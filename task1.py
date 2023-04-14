import pandas
import open3d as o3d
import matplotlib
import numpy as np
from open3d import visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    "Hip_Center",
    "Spine",
    "Shoulder_Center",
    "Head",
    "Shoulder_Left",
    "Elbow_Left",
    "Wrist_Left",
    "Hand_Left",
    "Shoulder_Right",
    "Elbow_Right",
    "Wrist_Right",
    "Hand_Right",
    "Hip_Left",
    "Knee_Left",
    "Ankle_Left",
    "Foot_Left",
    "Hip_Right",
    "Knee_Right",
    "Ankle_Right",
    "Foot_Right",
]


df = pandas.read_csv("train-final.csv")
df = df.drop("gesture label", axis=1)
df.fillna(df.mean())
df_norm = (df - df.mean()) / (df.max() - df.min())


class Joint:
    def __init__(self, name: str, xyz, ang, mean, std):
        self.name = name
        self.xyz = xyz
        self.ang = ang
        self.mean = mean
        self.std = std

    def move(self, other: "Joint"):
        self.xyz = [
            self.xyz[0] - other.xyz[0],
            self.xyz[1] - other.xyz[1],
            self.xyz[2] - other.xyz[2],
        ]

    def __str__(self) -> str:
        return f"""{self.name} : 
            xyz     : {self.xyz},
            angle   : {self.ang},
            mean    : {self.mean},
            std     : {self.std}
        """

    def __repr__(self) -> str:
        return self.__str__()


class Gesture:
    def __init__(self, name, joints: list[Joint]):
        self.name = name
        self.joints = joints
        self.pcl = None
        self.angles = []
        self.coords = []
        self.vecs = None

    def __str__(self) -> str:
        return f"""{self.name}
    Joints : {self.joints}
        """

    def __repr__(self) -> str:
        return self.__str__()

    def norm_pos(self):
        lower_back = [joint for joint in self.joints if joint.name == "Spine"][0]
        for joint in self.joints:
            if joint.name == "Spine":
                pass
            joint.move(lower_back)
        lower_back.move(lower_back)

    def to_pcl(self):
        positions = []
        angles = []
        for joint in self.joints:
            positions.append(joint.xyz)
            angles.append(joint.ang)
        self.angles = angles
        self.coords = positions
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.lables
        self.pcl = pcd

    def render_pcl(self):
        if not self.pcl:
            raise ValueError("You need to run <variable>.to_pcl() first")
        visualization.draw_geometries([self.pcl])

    def quiver(self):
        vecs = []
        for coord, angle in zip(self.coords, self.angles):
            vecs.append(coord)
            vecs[-1].extend(angle)
        self.vecs = vecs

    def render_quiver(self):
        if not self.vecs:
            raise ValueError("You need to run <variable>.quiver() first")

        soa = np.array(self.vecs)
        X, Y, Z, U, V, W = zip(*soa)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.quiver(X, Y, Z, U, V, W)
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
        gestures.append(Gesture(gesture[0], gesture[1:]))
        gestures[-1].norm_pos()
        gestures[-1].to_pcl()
        gestures[-1].quiver()

    return gestures


df = df.sort_values(by=["gesture id"])
print(df)
gestures = pack(df)

print(len(gestures))
#FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(
#    size=1, origin=[0,0,0])
visualization.draw_geometries([*[gesture.pcl for gesture in gestures[1:4]]])
