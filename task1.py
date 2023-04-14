import pandas
import open3d as o3d
import matplotlib
import numpy as np
from open3d import visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm as mag
from scipy.spatial.transform import Rotation as R
import copy
import matplotlib.pyplot as plt
from os import environ


def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"


if __name__ == "__main__":
    suppress_qt_warnings()

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


def render_labels():
    xyz = "xyz"
    angle = ["phi", "theta", "epsilon"]

    lables = ",".join(
        [
            ",".join([f"{joints[(i)//3]} {xyz[i%3]}" for i in range(0, 60)]),
            ",".join([f"{joints[(i)//3]} {angle[i%3]}" for i in range(0, 60)]),
            ",".join([f"mean {joints[(i)//3]} {angle[i%3]}" for i in range(0, 60)]),
            ",".join([f"std {joints[(i)//3]} {angle[i%3]}" for i in range(0, 60)]),
        ]
    )
    print(lables)
    exit(0)


# render_labels()


df = pandas.read_csv("train-final.csv")
df = df.drop("gesture label", axis=1)
df.fillna(df.mean())
df_norm = (df - df.mean()) / (df.max() - df.min())


class Joint:
    def __init__(self, name: str, xyz, ang, mean, std):
        self.name = name
        self.xyz = np.array(xyz)
        self.ang = ang
        self.mean = mean
        self.std = std

    def move(self, other: "Joint"):
        self.xyz = self.xyz - other.xyz

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
        self.joints_hash = {joint.name: joint for joint in self.joints}
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
        lower_back = [joint for joint in self.joints if joint.name == "Spine"][0]
        target = copy.deepcopy(lower_back)
        for joint in self.joints:
            joint.xyz = joint.xyz - target.xyz

    def torso_angle(self):

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
        phi = np.dot(n, k) / (mag(n))
        self.rotation = -phi

    def correct_rotation(self):
        """
        Rotates the entire figure by -phi
        """

        rot = R.from_rotvec([0, 0, self.rotation])
        for joint in self.joints:
            joint.xyz = rot.apply(joint.xyz)

    def to_pcl(self):
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
            color = (index + 1) / (len(lookup) + 1)
            color_joint = (joints.index(joint.name) + 1) / (len(joints) + 1)
            colors.append([color, 0, 1 - color])

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


def pack(data: pandas.DataFrame) -> list[Gesture]:
    gestures = []
    global joint
    for _, row in data.iterrows():
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


df = df.sort_values(by=["gesture id"])

gestures = pack(df)


print(len(gestures))
FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1 / 10, origin=[0, 0, 0])

nmbr_of_gestures = 2

visualization.draw_geometries(
    [
        FOR,
        *[gesture.pcl for gesture in gestures[0:nmbr_of_gestures]],
        *[gesture.ls for gesture in gestures[0:nmbr_of_gestures]],
    ]
)
# gestures[-1].matplot()
