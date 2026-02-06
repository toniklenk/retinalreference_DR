from typing import Type

import numpy as np

import vxpy.core.protocol as vxprotocol
from visuals.cmn_redesign import ContiguousMotionNoise3D, CMN3D20240410, CMN3D20240411
from visuals.spherical_global_motion import TranslationGrating, RotationGrating
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground

class CMN3DRotAndTrans_withEyemovements_20260211(vxprotocol.StaticProtocol):

    def create(self):
        axis = [(0, 0), (180, 0), (90, 0), (-90, 0), (0, 90), (0, 90)]

        """Translation"""
        for i in range(4):
            for azim, elev in axis:
                p = vxprotocol.Phase(duration=5)

                # angular_velocity = 0 -> pause
                p.set_visual(TranslationGrating,
                                  {TranslationGrating.azimuth: azim,
                                   TranslationGrating.elevation: elev,
                                   TranslationGrating.angular_period: 20,
                                   TranslationGrating.angular_velocity: 0})
                self.add_phase(p)

                # angular_velocity != 0 -> moving stimulus
                p = vxprotocol.Phase(duration=12)
                p.set_visual(TranslationGrating,
                                  {TranslationGrating.azimuth: azim,
                                   TranslationGrating.elevation: elev,
                                   TranslationGrating.angular_period: 20,
                                   TranslationGrating.angular_velocity: 30})
                self.add_phase(p)

        """Rotation"""
        for i in range(4):
            for azim, elev in axis:
                # angular_velocity = 0 -> pause
                p = vxprotocol.Phase(duration=5)
                p.set_visual(RotationGrating,
                             {RotationGrating.azimuth: azim,
                              RotationGrating.elevation: elev,
                              RotationGrating.angular_period: 20,
                              RotationGrating.angular_velocity: 0,
                              RotationGrating.waveform: 'rect'})
                self.add_phase(p)

                # angular_velocity != 0 -> moving stimulus
                p = vxprotocol.Phase(duration=12)
                p.set_visual(RotationGrating,
                             {RotationGrating.azimuth: azim,
                              RotationGrating.elevation: elev,
                              RotationGrating.angular_period: 20,
                              RotationGrating.angular_velocity: 30,
                              RotationGrating.waveform: 'rect'})
                self.add_phase(p)

        # TODO: longer pause here ?
        # Black
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(p)

        """Continuous Motion Noise"""
        # Grey
        gray_phase = vxprotocol.Phase(duration=10)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(gray_phase)

        # CMN
        phase = vxprotocol.Phase(duration=10 * 60)
        phase.set_visual(CMN3D20240606Vel140Scale7, {CMN3D20240606Vel140Scale7.reset_time: 1})
        self.add_phase(phase)

        # Black
        gray_phase = vxprotocol.Phase(duration=5 * 60)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(gray_phase)

        # CMN
        phase = vxprotocol.Phase(duration=10 * 60)
        phase.set_visual(CMN3D20240606Vel140Scale7, {CMN3D20240606Vel140Scale7.reset_time: 1})
        self.add_phase(phase)

        # Black
        gray_phase = vxprotocol.Phase(duration=5 * 60)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(gray_phase)

        # Grey
        gray_phase = vxprotocol.Phase(duration=10)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(gray_phase)

        # CMN
        phase = vxprotocol.Phase(duration=10 * 60)
        phase.set_visual(CMN3D20240606Vel140Scale7, {CMN3D20240606Vel140Scale7.reset_time: 0})
        self.add_phase(phase)

        # Grey
        gray_phase = vxprotocol.Phase(duration=10)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(gray_phase)
