import numpy as np

def project_to_local_2d_vectors(
        normals: np.ndarray,
        vectors: np.ndarray
) -> np.ndarray:
    """
        Parameters:
                normals: np.array
                    centers of CMN stimuli

                vectors: np.array
                    3D CMN motion vectors
    """

    def crossproduct(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Workaround because of NoReturn situation in numpy.cross"""
        return np.cross(v1, v2)

    vnorms = np.array([0, 0, 1]) - normals * np.dot(normals, np.array([0, 0, 1]))[:, None]
    vnorms /= np.linalg.norm(vnorms, axis=1)[:, None]

    hnorms = -crossproduct(vnorms, normals)
    hnorms /= np.linalg.norm(hnorms, axis=1)[:, None]

    vectors_2d = np.zeros((*vectors.shape[:2], 2))
    for i, v in enumerate(vectors):
        # Calculate 2d motion vectors in coordinate system defined by local horizontal and vertical norms
        motvecs_2d = np.array([np.sum(v * hnorms, axis=1),
                               np.sum(v * vnorms, axis=1)])
        vectors_2d[i] = motvecs_2d.T

    return vectors_2d


def FE_similarity(F, E):
    """
        Calculate similarity between F and E according to the definition in Zhang et. at. 2022
        Parameters:
            E: Estimated receptive field
            F: Optic flow field fitted to E
    """
    return ((np.sum(F*E, axis=1)/
            (np.linalg.norm(np.clip(E, 0.0000001, 1.), axis=1) *
             np.linalg.norm(F, axis=1)))
            .mean())

def tof(
        angle_azimuth: float,
        angle_elevation: float,
        P: np.ndarray):
    """
        Translational optic flow field.
        Returns a translational optic flow field for a given translation axis.

        Parameters:
            angle_azimuth (float):
                angle in radians of the azimuth of the translation axis.
            angle_elevation (float):
                angle in radians of the elevation of the translation axis.
            P (np.array):
                3D positions to sample flow field at.

        Returns:
            F (np.array):
                Return values of optic flow field (3D vector) at each point
                given in positions.
            FoC, FoE (np.array):
                Focus of Expansion and Focus of Contraction of the optic flow field.
                Given as 3D unit vector.
    """
    from numpy import sin, cos
    axis, a, b = np.array([1,0,0]), angle_azimuth, angle_elevation
    # yaw and pitch rotation matrices
    yaw=np.array([[cos(a),-sin(a),0],[sin(a), cos(a), 0],[0,0,1]])
    pitch=np.array([[cos(b), 0, sin(b)],[0,1,0],[-sin(b), 0, cos(b)]])
    T=axis @ yaw @ pitch
    return T - np.dot(P, T)[:,None] * P

def rof(
        angle_azimuth: float,
        angle_elevation: float,
        P: np.ndarray):
    """
        Rotational optic flow field.
        Returns a rotation optic flow field for a given translation axis.

        Parameters:
            angle_azimuth (float):
                angle in radians of the azimuth of the rotation axis.
            angle_elevation (float):
                angle in radians of the elevation of the rotation axis.
            P (np.array):
                3D positions to sample flow field at.

        Returns:
            F (np.array):
                Return values of optic flow field (3D vector) at each point
                given in positions.
            FoC, FoE (np.array):
                Focus of Expansion and Focus of Contraction of the optic flow field.
                Given as 3D unit vector.
    """
    from numpy import sin, cos
    axis, a, b = np.array([1,0,0]), angle_azimuth, angle_elevation
    # yaw and pitch rotation matrices
    yaw=np.array([[cos(a),-sin(a),0],[sin(a), cos(a), 0],[0,0,1]])
    pitch=np.array([[cos(b), 0, sin(b)],[0,1,0],[-sin(b), 0, cos(b)]])
    return -np.cross(P, axis @ yaw @ pitch)

def RSSangle(F, E):
    """
        Calculate RSS (residual sum of squared) of angles between two vector fields.
        Used in this analysis to calculate RSS between estimated RF and optic flow field,
        as an error function to fit the latter to the former.
        Parameters:
            F (np.array):
                Optic flow field to fit.
            E (np.array):
                Estimated RF.
        Returns:
            RSS (float)
    """
    #print(E)
    coefficients = np.sum(F*E, axis=1)/ (np.linalg.norm(np.clip(E, 0.0000001, 1.), axis=1)* np.linalg.norm(F, axis=1))
    angles = np.arccos(np.clip(coefficients, -1.0, 1.0))
    return np.sum(angles**2)

def RSSangle_Fto2D(F, E, pos):
    """
        Wraps around def RSSangle(F, E) to transform F from 3D to 2D,
        for convenience in using it with scipy.optimize.minimize
    """
    F=project_to_local_2d_vectors(pos, F[None,:,:]).squeeze()
    return RSSangle(F, E)
