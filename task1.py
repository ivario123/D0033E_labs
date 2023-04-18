import copy
from os import environ
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas
import seaborn as sn
from numpy.linalg import norm as mag
from open3d import visualization
from scipy.spatial.transform import Rotation as R


def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"


if __name__ == "__main__":
    suppress_qt_warnings()

parent_joints = {
    "Head": None,
    "Shoulder_Center": "Head",
    "Shoulder_Left": "Shoulder_Center",
    "Shoulder_Right": "Shoulder_Center",
    "Elbow_Left": "Shoulder_Left",
    "Elbow_Right": "Shoulder_Right",
    "Wrist_Left": "Elbow_Left",
    "Wrist_Right": "Elbow_Right",
    "Hand_Left": "Wrist_Left",
    "Hand_Right": "Wrist_Right",
    "Spine": "Shoulder_Center",
    "Hip_Center": "Spine",
    "Hip_Left": "Hip_Center",
    "Hip_Right": "Hip_Center",
    "Knee_Left": "Hip_Left",
    "Knee_Right": "Hip_Right",
    "Ankle_Left": "Knee_Left",
    "Ankle_Right": "Knee_Right",
    "Foot_Left": "Ankle_Left",
    "Foot_Right": "Ankle_Right",
}

lookup = [
    "afternoon",
    "baby",
    "big",
    "born",
    "bye",
    "calendar",
    "child",
    "cloud",
    "come",
    "daily",
    "dance",
    "dark",
    "day",
    "enjoy",
    "go",
    "hello",
    "home",
    "love",
    "my",
    "name",
    "no",
    "rain",
    "sorry",
    "strong",
    "study",
    "thankyou",
    "welcome",
    "wind",
    "yes",
    "you",
]


joints = [
    "Head",
    "Shoulder_Center",
    "Shoulder_Left",
    "Shoulder_Right",
    "Elbow_Left",
    "Elbow_Right",
    "Wrist_Left",
    "Wrist_Right",
    "Hand_Left",
    "Hand_Right",
    "Spine",
    "Hip_Center",
    "Hip_Left",
    "Hip_Right",
    "Knee_Left",
    "Knee_Right",
    "Ankle_Left",
    "Ankle_Right",
    "Foot_Left",
    "Foot_Right",
]
# Cutoff
blacklist_index = joints.index("Foot_Right") + 1


def render_labels(cap=len(joints)):
    xyz = "xyz"
    angle = ["phi", "theta", "epsilon"]
    r = range(0, cap * 3)
    labels = [
        *[f"{joints[(i)//3]} {xyz[i%3]}" for i in r],
        *[f"{joints[(i)//3]} {angle[i%3]}" for i in r],
        *[f"mean {joints[(i)//3]} {angle[i%3]}" for i in r],
        *[f"std {joints[(i)//3]} {angle[i%3]}" for i in r],
    ]
    return labels


# The labels can be generated using the following code snippet


def generate_labels_for_csv():
    labels = render_labels()
    labels.extend(["gesture label", "gesture id"])
    labels = ",".join(labels)
    print(labels)
    exit()


# generate_labels_for_csv()


class Joint:
    def __init__(self, name: str, xyz, ang, mean, std):
        self.name = name
        self.xyz = np.array(xyz)
        self.ang = ang
        self.mean = mean
        self.std = std

    def __str__(self) -> str:
        return f"""{self.name} : 
            xyz     : {self.xyz},
            angle   : {self.ang},
            mean    : {self.mean},
            std     : {self.std}
        """

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, __value: object) -> bool:

        if all(self.xyz == __value.xyz):
            return True
        return False


class Gesture:
    def __init__(self, name, joints: list[Joint]):
        self.name = name
        self.joints = joints
        self.joints_hash: dict[str, Joint] = {
            joint.name: joint for joint in self.joints
        }
        self.pcl = None
        self.angles = []
        self.coords = []
        self.vecs = None
        self.torso_angle()

    def __str__(self) -> str:
        return f"""{self.name}
    Joints : {self.joints}
        """

    def __repr__(self) -> str:
        return self.__str__()

    def norm_pos(self):
        """
        Moves the skeleton to the origin
        """
        lower_back = [joint for joint in self.joints if joint.name == "Spine"][0]
        target = copy.deepcopy(lower_back)
        for joint in self.joints:
            joint.xyz = joint.xyz - target.xyz

    def torso_angle(self):
        """
        Computes the angle of the torso and stores it
        """
        if self.name == "baby":
            i = 2
        relevant_joints = ["Shoulder_Left", "Shoulder_Right", "Spine"]
        relevant_joints = {
            joint.name: joint for joint in self.joints if joint.name in relevant_joints
        }
        spine, ls, rs = (
            relevant_joints["Spine"].xyz,
            relevant_joints["Shoulder_Left"].xyz,
            relevant_joints["Shoulder_Right"].xyz,
        )

        cl: np.ndarray = ls - spine
        cr: np.ndarray = rs - spine
        n = (1 / (mag(cl) * mag(cr))) * np.cross(cl, cr)
        k = np.array([0, 1, 0])

        if mag(n) != 0:
            phi = np.dot(n, k) / (mag(n))
        else:
            phi = 0
        self.rotation = -phi

    def correct_rotation(self):
        """
        Rotates the entire figure by -phi
        """
        rot = R.from_rotvec([0, 0, self.rotation])
        for joint in self.joints:
            joint.xyz = rot.apply(joint.xyz)

    def to_pcl(self):
        """
        Creates a point-cloud and the connecting vectors
        """
        self.positions = []
        angles = []
        colors = []
        indices = []
        for joint in self.joints:

            # Add the joint to the point cloud
            self.positions.append(joint.xyz)
            angles.append(joint.ang)
            index = lookup.index(self.name)

            # Add some color to the point cloud
            # color = (index + 1) / (len(lookup) + 1)
            if "Left" in joint.name:
                colors.append(
                    [1, (joints.index(joint.name) + 1) / (len(joints) + 1), 0]
                )
            elif "Right" in joint.name:
                colors.append(
                    [0, 1, (joints.index(joint.name) + 1) / (len(joints) + 1)]
                )
            else:
                colors.append(
                    [(joints.index(joint.name) + 1) / (len(joints) + 1), 0, 1]
                )

            # color_joint = (joints.index(joint.name) + 1) / (len(joints) + 1)
            # colors.append([color_joint, 0, 1 - color_joint * 0.2])

            # Draw a line to the parent joint
            if joint.name in parent_joints.keys():
                parent = parent_joints[joint.name]
                if not parent:
                    continue

                indices.append([joints.index(joint.name), joints.index(parent)])

        self.angles = angles
        self.coords = self.positions
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.positions)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(self.positions)
        ls.lines = o3d.utility.Vector2iVector(indices)
        ls.colors = o3d.utility.Vector3dVector(colors)
        self.pcl = pcd
        self.ls = ls

    def render_pcl(self):
        if not self.pcl:
            raise ValueError("You need to run <variable>.to_pcl() first")
        visualization.draw_geometries([self.pcl])

    def matplot(self):
        fig = plt.figure()
        sbplt = fig.add_subplot(projection="3d")
        sbplt.set_xlabel("X")
        sbplt.set_ylabel("Y")
        sbplt.set_zlabel("Z")
        for joint in self.joints:
            xyz = joint.xyz
            sbplt.scatter(*xyz, marker="o")
            sbplt.text(*xyz, joint.name)

        plt.show()

    def to_vec(self):
        map = {
            "xyz": [],
            "mean": [],
            "std": [],
            "ang": [],
        }
        global joints
        for joint in joints[:blacklist_index]:

            map["xyz"].extend(self.joints_hash[joint].xyz)
            map["mean"].extend(self.joints_hash[joint].mean)
            map["std"].extend(self.joints_hash[joint].std)
            map["ang"].extend(self.joints_hash[joint].ang)
        ret = []

        for key in map.keys():
            ret.extend(map[key])
        ret.append(int(lookup.index(self.name)))
        return ret


def to_df(gestures: List[Gesture]):
    data = [gesture.to_vec() for gesture in gestures]
    labels = render_labels(cap=blacklist_index)
    labels.append("gesture id")
    data = np.array(data)
    df = pandas.DataFrame(data, columns=labels)
    return df


def remove_correlated(df: pandas.DataFrame):
    """
    Removes all columns that have atleast 98% correlation with another column
    """
    original_shape = df.shape
    corr = df.corr().abs()  # find correlation
    # Create a heatmap
    sn.heatmap(corr)
    print()
    print("=" * 20)
    print("Showing correlation matrix")
    print("=" * 20)
    print()

    limit = 0.95
    to_remove = []
    explanation = []
    for i in corr.columns:
        corr[i][i] = 0
    for i, col in enumerate(corr.columns):
        if col in to_remove:
            continue
        for id, el in enumerate(corr[col]):
            if id == i:
                continue
            if el > limit:
                explanation.append(
                    f"{col} has correlation of {el} with {list(corr.columns)[id]} and will be removed"
                )
                # print(i, id, el)
                to_remove.append(col)
                break
    print("=" * 20, "\n" * 2)
    print(f"\n\n{'-'*20}\n\n".join(explanation))
    print("\n" * 2, "=" * 20)
    # Borrowed from https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
    df.drop(to_remove, axis=1, inplace=True)

    plt.show()
    reduced_shape = df.shape
    print(f"{original_shape=} {reduced_shape=}")
    return df


def pack(data: pandas.DataFrame) -> list[Gesture]:
    gestures = []
    global joint
    for id, (_, row) in enumerate(data.iterrows()):
        gesture = [lookup[int(row.get("gesture id") - 1)]]
        for joint in joints:
            xyz = [row.get(f"{joint} {id}") for id in "xyz"]
            ang = [row.get(f"{joint} {id}") for id in ["phi", "theta", "epsilon"]]
            mean = [row.get(f"mean {joint} {id}") for id in ["phi", "theta", "epsilon"]]
            std = [row.get(f"std {joint} {id}") for id in ["phi", "theta", "epsilon"]]
            joint = Joint(joint, xyz, ang, mean, std)
            gesture.append(joint)
        gesture = Gesture(gesture[0], gesture[1:])
        gesture.norm_pos()
        gesture.correct_rotation()
        gesture.to_pcl()
        gestures.append(gesture)

    return gestures


def visualize_gesture(id, gestures, hash_gestures):
    FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1 / 10, origin=[0, 0, 0]
    )
    if type(id) == int:
        visualization.draw_geometries(
            [
                FOR,
                gestures[id].pcl,
                gestures[id].ls,
            ],
            window_name=f"gesture id {id}",
        )
    elif type(id) == List[Gesture]:
        visualization.draw_geometries(
            [
                FOR,
                *[gesture.pcl for gesture in id],
                *[gesture.ls for gesture in id],
            ],
            window_name=f"User specified gestures",
        )
    elif type(id) == range:
        visualization.draw_geometries(
            [
                FOR,
                *[gestures[index].pcl for index in id],
                *[gestures[index].ls for index in id],
            ],
            window_name=f"Gestures in range {id}",
        )
    elif type(id) == str:
        visualization.draw_geometries(
            [
                FOR,
                *[gesture.pcl for gesture in hash_gestures[id]],
                *[gesture.ls for gesture in hash_gestures[id]],
            ],
            window_name=f"all {id} gestures",
        )


def cleanup_data_and_save(df: pandas.DataFrame):
    df = to_df(gestures)
    df = remove_correlated(df)
    df.to_csv("data_pp.csv", index=False)


# Read data from the csv using pandas
df = pandas.read_csv("train-final.csv")

# We don't need the text version of the label
df = df.drop("gesture label", axis=1)


# Replace missing values with mean of that column
df = df.fillna(df.mean())


# Pack all of the csv into python objects for visualization and pre-processing
gestures = pack(df)

# Convert the gesture list to a dict for ease of use
hash_gestures = {
    gesture_id: [gesture for gesture in gestures if gesture.name == gesture_id]
    for gesture_id in lookup
}


# Visualize the data in a few ways
visualize_gesture(range(0, 10), gestures, hash_gestures)
visualize_gesture(0, gestures, hash_gestures)
visualize_gesture("thankyou", gestures, hash_gestures)
gestures[-1].matplot()

"""
    Clean up and store result in data_pp.csv
"""
cleanup_data_and_save(df)
