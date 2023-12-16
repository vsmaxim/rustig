use nalgebra::Point3;


pub struct Player {
    pub position: Point3<f32>,
    speed: f32,
}

impl Player {
    pub fn new(speed: f32) -> Self {
        Self { 
            position: Point3::new(2.0, 2.0, 2.0),
            speed,
        }
    }

    fn move_position(
        &mut self,
        dx: f32,
        dy: f32,
        dz: f32,
    ) {
        self.position[0] += dx;
        self.position[1] += dy;
        self.position[2] += dz;
        println!("Player position: {}", self.position);
    }

    pub fn move_left(&mut self) {
        self.move_position(-self.speed, 0.0, 0.0);
    }

    pub fn move_right(&mut self) {
        self.move_position(self.speed, 0.0, 0.0);
    }

    pub fn move_forward(&mut self) {
        self.move_position(0.0, 0.0, self.speed);
    }

    pub fn move_back(&mut self) {
        self.move_position(0.0, 0.0, -self.speed);
    }
}
