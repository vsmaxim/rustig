use std::sync::Arc;

use device::GameDevice;
use graphics::ApplicationView;
use window::Application;
use winit::event_loop::EventLoop;

mod graphics;
mod window;
mod device;
mod ubo;
mod state;
mod camera;
mod player;
mod object;
mod vertex;


fn main() {
    let event_loop = EventLoop::new();

    let device = Arc::new(GameDevice::new(&event_loop, true));
    let view = ApplicationView::new(&device);

    Application::new(device, view)
        .run(event_loop);
}

