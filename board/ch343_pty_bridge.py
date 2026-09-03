#!/usr/bin/env python3
"""Userspace CH343 (WCH USB-serial) -> PTY bridge.

The board image ships no USB-serial kernel driver (no cdc_acm / ch341), so the
SO-101's motor adapter never shows up as /dev/ttyACM*. This script drives the
CH343 (a CDC-ACM device) from userspace via libusb/pyusb and exposes it as a
pseudo-terminal symlinked to a stable name (default /dev/ttyFOLLOWER).
Point LeRobot's `port` at that symlink; LeRobot stays stock.

Requires: libusb-1.0 (in the board image) + pyusb (in the board venv).
Run:  "$BOARD_PY" ch343_pty_bridge.py &
"""
import argparse, errno, os, pty, signal, sys, termios, threading, tty, struct
import usb.core, usb.util

try:
    USBTimeout = usb.core.USBTimeoutError
except AttributeError:                      # older pyusb
    USBTimeout = usb.core.USBError

# ---- termios baud constant <-> integer, built from whatever this libc defines ----
BAUD_MAP = {}
for name in dir(termios):
    if name.startswith("B") and name[1:].isdigit():
        BAUD_MAP[getattr(termios, name)] = int(name[1:])

def cdc_set_line_coding(dev, comm_ifnum, baud, cflag):
    databits = {termios.CS5: 5, termios.CS6: 6, termios.CS7: 7, termios.CS8: 8}.get(cflag & termios.CSIZE, 8)
    if not (cflag & termios.PARENB):
        parity = 0
    else:
        parity = 1 if (cflag & termios.PARODD) else 2
    stop = 2 if (cflag & termios.CSTOPB) else 0          # CDC: 0=1 stop, 2=2 stop
    payload = struct.pack("<IBBB", baud, stop, parity, databits)
    # bmRequestType 0x21 = host->device | class | interface ; bRequest 0x20 = SET_LINE_CODING
    dev.ctrl_transfer(0x21, 0x20, 0, comm_ifnum, payload)

def cdc_set_control_line_state(dev, comm_ifnum, dtr=True, rts=True):
    # bRequest 0x22 = SET_CONTROL_LINE_STATE ; wValue bit0=DTR bit1=RTS (what pyserial asserts on open)
    val = (0x01 if dtr else 0) | (0x02 if rts else 0)
    dev.ctrl_transfer(0x21, 0x22, val, comm_ifnum, None)

def find_endpoints(dev):
    cfg = dev.get_active_configuration()
    ep_in = ep_out = data_if = comm_if = None
    for intf in cfg:
        if intf.bInterfaceClass == 0x02:                # CDC comm/ACM
            comm_if = intf.bInterfaceNumber
        for ep in intf:
            if usb.util.endpoint_type(ep.bmAttributes) == usb.util.ENDPOINT_TYPE_BULK:
                data_if = intf.bInterfaceNumber
                if usb.util.endpoint_direction(ep.bEndpointAddress) == usb.util.ENDPOINT_IN:
                    ep_in = ep
                else:
                    ep_out = ep
    if comm_if is None:
        comm_if = 0
    return comm_if, data_if, ep_in, ep_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid", type=lambda x: int(x, 16), default=0x1a86)
    ap.add_argument("--pid", type=lambda x: int(x, 16), default=0x55d3)
    ap.add_argument("--link", default="/dev/ttyFOLLOWER")
    ap.add_argument("--baud", type=int, default=1000000, help="initial baud (Feetech STS = 1e6)")
    args = ap.parse_args()

    dev = usb.core.find(idVendor=args.vid, idProduct=args.pid)
    if dev is None:
        sys.exit("device %04x:%04x not found" % (args.vid, args.pid))

    comm_if, data_if, ep_in, ep_out = find_endpoints(dev)
    if ep_in is None or ep_out is None:
        sys.exit("no bulk IN/OUT endpoints found")

    for ifnum in {comm_if, data_if}:
        if dev.is_kernel_driver_active(ifnum):
            dev.detach_kernel_driver(ifnum)
    usb.util.claim_interface(dev, data_if)
    cdc_set_line_coding(dev, comm_if, args.baud, termios.CS8)
    try:
        cdc_set_control_line_state(dev, comm_if, dtr=True, rts=True)
    except usb.core.USBError as e:
        print("warn: SET_CONTROL_LINE_STATE not supported (%s); continuing" % e)

    master_fd, slave_fd = pty.openpty()
    tty.setraw(slave_fd)                                  # binary-clean line discipline
    # Make the PTY's reported line speed match the initial USB baud, so the
    # termios auto-tracker below does NOT immediately downshift the device to
    # the pty default (38400). If this baud has no termios B-constant (e.g.
    # 250000), disable tracking and rely solely on --baud.
    initial_bspd = getattr(termios, "B%d" % args.baud, None)
    track = initial_bspd is not None
    if track:
        a = termios.tcgetattr(slave_fd)
        a[4] = a[5] = initial_bspd                        # ispeed / ospeed
        termios.tcsetattr(slave_fd, termios.TCSANOW, a)
    slave_path = os.ttyname(slave_fd)
    try:
        if os.path.islink(args.link) or os.path.exists(args.link):
            os.remove(args.link)
        os.symlink(slave_path, args.link)
    except OSError as e:
        print("warn: could not create symlink %s (%s); use %s directly" % (args.link, e, slave_path))

    print("BRIDGE UP: %04x:%04x  data_if=%d comm_if=%d  ep_in=0x%02x ep_out=0x%02x"
          % (args.vid, args.pid, data_if, comm_if, ep_in.bEndpointAddress, ep_out.bEndpointAddress))
    print("           port = %s  ->  %s   (baud %d)" % (args.link, slave_path, args.baud))
    sys.stdout.flush()

    stop = threading.Event()
    state = {"baud": args.baud, "cflag": termios.CS8}

    def tx():                                            # PTY -> USB (host writes to servos)
        import select
        while not stop.is_set():
            r, _, _ = select.select([master_fd], [], [], 0.2)
            if r:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    continue
                if data:
                    try:
                        ep_out.write(data, timeout=1000)
                    except usb.core.USBError as e:
                        print("tx USBError:", e)
            # cheap: re-apply line coding if client changed baud/format
            if track:
                try:
                    a = termios.tcgetattr(slave_fd)
                    baud = BAUD_MAP.get(a[5], state["baud"]); cflag = a[2]
                    if baud != state["baud"] or cflag != state["cflag"]:
                        state["baud"], state["cflag"] = baud, cflag
                        cdc_set_line_coding(dev, comm_if, baud, cflag)
                        print("line coding -> baud=%d" % baud)
                except Exception:
                    pass

    def rx():                                            # USB -> PTY (servo replies)
        n = ep_in.wMaxPacketSize * 8
        while not stop.is_set():
            try:
                data = ep_in.read(n, timeout=1000)
            except USBTimeout:
                continue
            except usb.core.USBError as e:
                if e.errno == errno.ENODEV:                  # adapter unplugged
                    print("device gone")
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
                if not stop.is_set():
                    print("rx USBError:", e)
                continue
            if len(data):
                try:
                    os.write(master_fd, bytes(data))
                except OSError:
                    pass

    def shutdown(*_):
        stop.set()
        try: os.remove(args.link)
        except OSError: pass
        try: usb.util.release_interface(dev, data_if)
        except Exception: pass
        print("bridge down")
        os._exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    ta = threading.Thread(target=tx, daemon=True); ta.start()
    tb = threading.Thread(target=rx, daemon=True); tb.start()
    signal.pause()

if __name__ == "__main__":
    main()
