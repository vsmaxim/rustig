use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};


#[derive(Copy, Clone, BufferContents, Vertex)]
#[repr(C)]
pub struct MyVertex {
    #[name("inPosition")]
    #[format(R32G32B32_SFLOAT)]
    pub pos: [f32; 3],

    #[name("inColor")]
    #[format(R32G32B32_SFLOAT)]
    pub col: [f32; 3],

    #[name("inTexCoord")]
    #[format(R32G32_SFLOAT)]
    pub tex_coord: [f32; 2]
}
