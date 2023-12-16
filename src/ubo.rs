use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use nalgebra::{Matrix4, Unit, Vector3, Point3};


#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>
}

fn to_radians(a: f32) -> f32 {
    a * std::f32::consts::PI / 180.0
}

impl UniformBufferObject {
    pub fn new(
        aspect: f32,
        camera_position: &Point3<f32>,
    ) -> Self {
        let model = Matrix4::from_axis_angle(
            &Unit::new_normalize(Vector3::new(1.0, 1.0, 0.0)),
            0.0,
        );

        let view = Matrix4::look_at_rh(
            camera_position,
            &Point3::new(0.0, 0.0, 0.0),
            &Vector3::new(0.0, 1.0, 0.0),
        );

        let proj = Matrix4::new_perspective(
            aspect,
            to_radians(45.0),
            0.1,
            10.0,
        );

        Self { model, view, proj }
    }
}
