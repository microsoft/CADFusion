import json
import pickle

SKETCH_R = 1
RADIUS_R = 1
EXTRUDE_R = 1.0
SCALE_R = 1.4
OFFSET_R = 0.9
PIX_PAD = 4
CMD_PAD = 3
COORD_PAD = 4
EXT_PAD = 1
EXTRA_PAD = 1
R_PAD = 2

def create_curve_str(se_xy, se_cmd):
    curve_str = ""
    xy_offset = 0
    if se_cmd == 0:  # line
        curve_str = " line," + ",".join(str(x) for x in se_xy[0])
        xy_offset = 2
    elif se_cmd == 1:  # arc
        curve_str = " arc," + ",".join(str(x) for x in se_xy[0:2].flatten())
        xy_offset = 3
    elif se_cmd == 2:  # circle
        curve_str = " circle," + ",".join(str(x) for x in se_xy[0:4].flatten())
        xy_offset = 5
    curve_str += " <curve_end>"
    return curve_str, xy_offset


def create_sketch_str(se_xy, se_cmd):
    sketch_str = ""
    len_xy, len_cmd = len(se_xy), len(se_cmd)
    xy_idx = 0
    for cmd_item in se_cmd:  # for each command
        if 0 <= cmd_item <= 2:  # curve
            curve_str, xy_offset = create_curve_str(se_xy[xy_idx:], cmd_item)
            sketch_str += curve_str
            xy_idx += xy_offset
        elif cmd_item == -1:  # loop
            sketch_str += " <loop_end>"
            xy_idx += 1
        elif cmd_item == -2:  # face
            sketch_str += " <face_end>"
            xy_idx += 1
        elif cmd_item == -3:  # sketch
            sketch_str += " <sketch_end>"
            xy_idx += 1
        else:
            raise ValueError("Invalid command: " + str(cmd_item))
    if xy_idx != len_xy:
        raise ValueError("xy_idx != len_xy")
    return sketch_str


def create_extrude_str(se_ext):
    extrude_str = ""
    # extrude operation
    if se_ext[14] == 1:
        extrude_str += "add"
    elif se_ext[14] == 2:
        extrude_str += "cut"
    elif se_ext[14] == 3:
        extrude_str += "intersect"
    else:
        raise ValueError("Invalid extrude operation: " + str(se_ext[14]))
    # other extrude parameters
    extrude_str = (
        extrude_str + "," + ",".join(str(x - EXT_PAD) for x in se_ext[0:5])
    )  # ext_v, ext_T
    extrude_str = (
        extrude_str + "," + ",".join(str(x - R_PAD) for x in se_ext[5:14])
    )  # ext_R
    extrude_str = (
        extrude_str + "," + ",".join(str(x - EXT_PAD) for x in se_ext[15:18])
    )  # scale, offset
    # extrude end
    extrude_str += " <extrude_end>"
    return extrude_str

def create_command_sequence(item):
    se_str = ""
    num_se = item["num_se"]
    for se_idx in range(num_se):  # for each sketch-extrude
        xy, cmd, ext = (
            item["se_xy"][se_idx] - COORD_PAD,
            item["se_cmd"][se_idx] - CMD_PAD,
            item["se_ext"][se_idx],
        )
        se_str = se_str + " " + create_sketch_str(xy, cmd).strip()
        se_str = se_str + " " + create_extrude_str(ext).strip()
    return se_str.strip()

with open("data/sl_data/train.json", "r") as f:
    train_data = json.load(f)
with open("data/sl_data/test.json", "r") as f:
    test_data = json.load(f)
with open("data/sl_data/val.json", "r") as f:
    val_data = json.load(f)

with open("cad_data/train_deduplicate_s.pkl", "rb") as f:
    sk_data = pickle.load(f)

for item in train_data:
    serial_num = item['serial_num']
    description = item['description']
    item["command_sequence"] = create_command_sequence(sk_data[serial_num])

for item in test_data:
    serial_num = item['serial_num']
    description = item['description']
    item["command_sequence"] = create_command_sequence(sk_data[serial_num])

for item in val_data:
    serial_num = item['serial_num']
    description = item['description']
    item["command_sequence"] = create_command_sequence(sk_data[serial_num])

with open("data/sl_data/train.json", "w+") as f:
    json.dump(train_data, f, indent=4)
with open("data/sl_data/test.json", "w+") as f:
    json.dump(test_data, f, indent=4)
with open("data/sl_data/val.json", "w+") as f:
    json.dump(val_data, f, indent=4)