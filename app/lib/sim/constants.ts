/** Kelana Jaya Line train: 4 cars, 800 people (AERA's capacity), 200 per car, two cameras per car. */
export const CARS = 4;
export const TRAIN_CAPACITY = 800;
export const CAR_CAPACITY = TRAIN_CAPACITY / CARS;
export const CAMERAS_PER_CAR = 2;
export const VIEW_CAPACITY = CAR_CAPACITY / CAMERAS_PER_CAR;
export const FPS = 2; // the camera counter's rate in AERA's edge code
