import pandas
import open3d as o3d
import matplotlib
from open3d import visualization


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

    def __str__(self) -> str:
        return f"""{self.name}
    Joints : {self.joints}
        """

    def __repr__(self) -> str:
        return self.__str__()

    def to_pcl(self):
        positions = []
        angles = []
        for joint in self.joints:
            positions.append(joint.xyz)
            angles = []

        self.coords = positions
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.an
        self.pcl = pcd

    def show(self):
        if not self.pcl:
            raise ValueError("You need to run <variable>.to_pcl() first")
        visualization.draw_geometries([self.pcl])
    
    def quiver(self):


        pass



def pack(data: pandas.DataFrame):
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
        gestures[-1].to_pcl()
    return gestures

print(df)
gestures = pack(df)
gestures[-1].show()
