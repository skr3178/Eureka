class AntEnv:
    """Rest of the environment definition omitted."""
    
    """
    ### Observation Space
    
    Observations consist of positional values of different body parts of the Ant,
    followed by the velocities of those individual parts (their derivatives) with all the
    positions ordered before all the velocities.
    
    By default, the observation does not include the x- and y-coordinates of the torso.
    These can be included by passing `exclude_current_positions_from_observation=False` during construction.
    In this case, the observation space will be a `Box(-Inf, Inf, (107,), float64)`, where the first two observations are the x- and y-coordinates of the torso.
    
    By default, however, the observation space is a `Box(-Inf, Inf, (105,), float64)`, where the position and velocity elements are as follows:
    
    | Num | Observation                                                  | Min    | Max    | Name (in corresponding XML file)       | Joint | Type (Unit)              |
    |-----|--------------------------------------------------------------|--------|--------|----------------------------------------|-------|--------------------------|
    | 0   | z-coordinate of the torso (centre)                           | -Inf   | Inf    | root                                   | free  | position (m)             |
    | 1   | w-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
    | 2   | x-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
    | 3   | y-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
    | 4   | z-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
    | 5   | angle between torso and first link on front left             | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
    | 6   | angle between the two links on the front left                | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
    | 7   | angle between torso and first link on front right            | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
    | 8   | angle between the two links on the front right               | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
    | 9   | angle between torso and first link on back left              | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
    | 10  | angle between the two links on the back left                 | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
    | 11  | angle between torso and first link on back right             | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
    | 12  | angle between the two links on the back right                | -Inf   | Inf    | ankle_4 (right_back_leg)               | hinge | angle (rad)              |
    | 13  | x-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
    | 14  | y-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
    | 15  | z-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
    | 16  | x-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
    | 17  | y-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
    | 18  | z-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
    | 19  | angular velocity of angle between torso and front left link  | -Inf   | Inf    | hip_1 (front_left_leg)                 | hinge | angle (rad)              |
    | 20  | angular velocity of the angle between front left links       | -Inf   | Inf    | ankle_1 (front_left_leg)               | hinge | angle (rad)              |
    | 21  | angular velocity of angle between torso and front right link | -Inf   | Inf    | hip_2 (front_right_leg)                | hinge | angle (rad)              |
    | 22  | angular velocity of the angle between front right links      | -Inf   | Inf    | ankle_2 (front_right_leg)              | hinge | angle (rad)              |
    | 23  | angular velocity of angle between torso and back left link   | -Inf   | Inf    | hip_3 (back_leg)                       | hinge | angle (rad)              |
    | 24  | angular velocity of the angle between back left links        | -Inf   | Inf    | ankle_3 (back_leg)                     | hinge | angle (rad)              |
    | 25  | angular velocity of angle between torso and back right link  | -Inf   | Inf    | hip_4 (right_back_leg)                 | hinge | angle (rad)              |
    | 26  | angular velocity of the angle between back right links       | -Inf   | Inf    | ankle_4 (right_back_leg)               | hinge | angle (rad)              |
    | excluded | x-coordinate of the torso (centre)                      | -Inf   | Inf    | root                                   | free  | position (m)             |
    | excluded | y-coordinate of the torso (centre)                      | -Inf   | Inf    | root                                   | free  | position (m)             |
    
    Additionally, after all the positional and velocity based values in the table,
    the observation contains (in order):
    
    - *cfrc_ext:* This is the center of mass based external forces on the body parts.
      It has shape 13 * 6 (*nbody * 6*) and hence adds another 78 elements to the state space.
      (external forces - force x, y, z and torque x, y, z)
      Note: The cfrc_ext of worldbody (index 0) is excluded as it is always 0.
    
    The body parts are:
    
    | body part                 | id (for `v5`) |
    |  -----------------------  |  ---   |
    | worldbody (note: all values are constant 0) | excluded |
    | torso                     | 0       |
    | front_left_leg            | 1       |
    | aux_1 (front left leg)    | 2       |
    | ankle_1 (front left leg)  | 3       |
    | front_right_leg           | 4       |
    | aux_2 (front right leg)   | 5       |
    | ankle_2 (front right leg) | 6       |
    | back_leg (back left leg)  | 7       |
    | aux_3 (back left leg)     | 8       |
    | ankle_3 (back left leg)   | 9       |
    | right_back_leg            | 10      |
    | aux_4 (back right leg)    | 11      |
    | ankle_4 (back right leg)  | 12      |
    
    The (x,y,z) coordinates are translational DOFs, while the orientations are rotational DOFs expressed as quaternions.
    
    Note that the obs argument passed to the reward function is a concatenation of obs across
    all environments, so it will have shape (n, 105) by default (or (n, 107) if x,y positions are included).
    """
    
    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        
        if self._exclude_current_positions_from_observation:
            position = position[2:]
        
        if self._include_cfrc_ext_in_observation:
            contact_force = self.contact_forces[1:].flatten()
            return np.concatenate((position, velocity, contact_force))
        else:
            return np.concatenate((position, velocity))

