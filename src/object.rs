use crate::vertex::MyVertex;


pub struct RenderableObject {
    pub vertices: Vec<MyVertex>,
    pub indices: Vec<u32>,
}


impl RenderableObject {
    pub fn load(path: &str) -> Self {
        let (vertices, indices) = Self::load_object(path);
        Self { vertices, indices }
    }

    fn load_object(path: &str) -> (Vec<MyVertex>, Vec<u32>) {
        let (models, _) = tobj::load_obj(
            path,
            &tobj::LoadOptions {
                single_index: true,
                ..Default::default()
            },
        ).expect("Failed to load object file");

        let mut vertices = Vec::<MyVertex>::new();
        let mut indices = Vec::<u32>::new();

        for m in models.iter() {
            let mesh = &m.mesh;

            for index in 0..(mesh.positions.len() / 3) {
                vertices.push(MyVertex {
                    pos: [
                        mesh.positions[(index * 3) as usize],
                        mesh.positions[(index * 3 + 1) as usize],
                        mesh.positions[(index * 3 + 2) as usize],
                    ],
                    col: [1.0, 1.0, 1.0],
                    tex_coord: [
                        mesh.texcoords[(index * 2) as usize],
                        1.0 - mesh.texcoords[(index * 2 + 1) as usize],
                    ],
                });
            }

            indices.clone_from(&mesh.indices);
        }

        println!("loaded {} vertices, {} indices", vertices.len(), indices.len());
        (vertices, indices)
    }
}
