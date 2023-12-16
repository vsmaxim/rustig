use nalgebra::{Point3, Vector3};


pub struct Camera {
    camera_diff: Vector3<f32>,
    view_diff: Vector3<f32>,
    pub position: Point3<f32>,
    pub look_at: Point3<f32>,
}

const CAMERA_DIFF: Vector3<f32> = Vector3::<f32>::new(-2.0, 0.0, 1.5);
const VIEW_DIFF: Vector3<f32> = Vector3::<f32>::new(1.0, 0.3, 0.0);


impl Camera {
    pub fn new(player_position: &Point3<f32>) -> Self {
        let initial_position = Self::calc_camera_position(
            player_position,
            &CAMERA_DIFF
        ); 

        let initial_look_at = Self::calc_look_at_position(
            player_position, 
            &VIEW_DIFF,
        );

        Self {
            camera_diff: CAMERA_DIFF,
            view_diff: VIEW_DIFF,
            position: initial_position.clone(),
            look_at: initial_look_at,
        }
    }

    pub fn move_camera(&mut self, position: &Point3<f32>) {
        self.position = Self::calc_camera_position(position, &self.camera_diff);
        self.look_at = Self::calc_look_at_position(position, &self.view_diff);

        println!("Camera position: {}", self.position);
        println!("Camera look at: {}", self.look_at);
    }

    fn calc_camera_position(
        player_position: &Point3<f32>,
        camera_diff: &Vector3<f32>,
    ) -> Point3<f32> {
        player_position + camera_diff
    }

    fn calc_look_at_position(
        player_position: &Point3<f32>,
        view_diff: &Vector3<f32>,
    ) -> Point3<f32> {
        player_position + view_diff
    }
}
