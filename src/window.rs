use std::sync::Arc;

use nalgebra::{Point3, Vector3};
use winit::{event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState}, event_loop::EventLoop};
use crate::{device::GameDevice, graphics::ApplicationView, camera::Camera, player::Player, object::RenderableObject};


pub trait SceneView {
    fn init_view(&mut self);
    fn draw_frame(&mut self, extent: [u32; 2]); 
    fn window_resized(&mut self);
}

pub struct Application {
    device: Arc<GameDevice>,
    view: ApplicationView,
    player: Player,
    camera: Camera,
    object_loaded: bool,
    load_delta: Vector3<f32>,
}

impl Application {
    pub fn new(device: Arc<GameDevice>, view: ApplicationView) -> Self {
        let player = Player::new(0.1);
        let camera = Camera::new(&player.position);
        let load_delta = Vector3::new(0.0, 0.0, 0.0);

        Self { 
            device,
            view,
            player,
            camera,
            object_loaded: false,
            load_delta,
        }
    }

    pub fn handle_keypress(&mut self, keycode: VirtualKeyCode) {
        let mut player_moved = false;

        match keycode {
            VirtualKeyCode::Left => {
                self.player.move_left();
                player_moved = true;
            }
            VirtualKeyCode::Right => {
                self.player.move_right();
                player_moved = true;
            }
            VirtualKeyCode::Up => {
                self.player.move_forward();
                player_moved = true;
            }
            VirtualKeyCode::Down => {
                self.player.move_back();
                player_moved = true;
            }
            VirtualKeyCode::L => {
                let obj = RenderableObject::load("src/models/viking_room.obj");
                self.view.render_object(obj, &self.load_delta);
                self.load_delta[0] += 0.5;
            }

            _ => ()
        }

        if player_moved {  
            self.camera.move_camera(&self.player.position);
            self.update_camera();
        }
    }

    pub fn update_camera(&mut self) {
        self.view.camera_position = self.camera.position;
        self.view.camera_look_at = self.camera.look_at;
        ()
    }

    pub fn run(mut self, event_loop: EventLoop<()>) {
        self.view.init_view();

        event_loop.run(move |event, _, control_flow| {
            control_flow.set_poll();

            match event {
                Event::WindowEvent { 
                    event: WindowEvent::CloseRequested, 
                    ..
                } => {
                    println!("Exiting...");
                    control_flow.set_exit();
                }

                Event::WindowEvent { 
                    event: WindowEvent::Resized(_),
                    ..
                } => {
                    self.view.window_resized();
                }

                Event::WindowEvent { 
                    event: WindowEvent::KeyboardInput {  
                        input: KeyboardInput { 
                            state: ElementState::Pressed,
                            virtual_keycode: Some(keycode),
                            ..
                        },
                        ..
                    },
                    ..
                } => {
                    println!("Key pressed: {:?}", keycode);
                    self.handle_keypress(keycode);
                }

                Event::MainEventsCleared => {
                    self.view.draw_frame(self.device.current_extent())
                }

                _ => ()
            }
        });
    }
}
