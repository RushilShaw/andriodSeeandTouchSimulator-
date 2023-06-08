from ppadb.client import Client


def get_hu_device() -> Client.DEVICE:
    """
    This function requires ADB to be running or else a RuntimeError error will arise
    This function gets the hu_device and returns it. If the device is not found then it will return None
    :return: Client.DEVICE
    """
    client = Client()
    devices = client.devices()
    client.devices()
    hu_device = None
    for device in devices:
        if "Harman" in device.shell("getprop ro.product.manufacturer"):
            hu_device = device
            break
    return hu_device

def record_touch_inputs(device):
    touch_records = []
    device.shell('getevent -lt /dev/input/event1', handler=lambda event: touch_records.append(event), read_timeout=0.1)

    for record in touch_records:
        event = record.split()
        if 'ABS_MT_POSITION_X' in event and 'ABS_MT_POSITION_Y' in event:
            x = int(event[event.index('ABS_MT_POSITION_X') + 1], 16)
            y = int(event[event.index('ABS_MT_POSITION_Y') + 1], 16)
            print(f"Touch input: x={x}, y={y}")


def main():
    hu_device: Client.DEVICE = get_hu_device()
    record_touch_inputs(hu_device)


if __name__ == '__main__':
    main()
