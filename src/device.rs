use std::sync::Arc;

use vulkano::{instance::{Instance, InstanceCreateInfo, debug::{DebugUtilsMessenger, DebugUtilsMessengerCreateInfo}, InstanceExtensions}, swapchain::{Surface, SurfaceInfo, SurfaceCapabilities}, VulkanLibrary, device::{physical::PhysicalDevice, QueueFlags, Features, DeviceExtensions}};
use vulkano_win::VkSurfaceBuild;
use winit::{event_loop::EventLoop, dpi::LogicalSize, window::WindowBuilder};


const WIDTH: u32 = 1200;
const HEIGHT: u32 = 1000;
const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];

pub struct QueueFamilyIndices {
    pub graphics_family: i32,
    pub present_family: i32,
}

impl QueueFamilyIndices {
    fn new() -> Self {
        Self { graphics_family: -1, present_family: -1 }
    }

    fn is_complete(&self) -> bool {
        self.graphics_family >= 0 && self.present_family >= 0
    }
}

pub struct GameDevice {
    pub instance: Arc<Instance>,
    pub surface: Arc<Surface>,
    device_index: usize,
}

impl GameDevice {
    pub fn new(event_loop: &EventLoop<()>, enable_validation: bool) -> Self {
        let logical_size = LogicalSize::new(
            f64::from(WIDTH),
            f64::from(HEIGHT),
        );

        let library = VulkanLibrary::new().unwrap();

        if enable_validation {
            Self::ensure_validation_supported(&library);
        }

        let app_info = InstanceCreateInfo{
            enumerate_portability: true,
            enabled_extensions: Self::get_required_extensions(&library, enable_validation),
            ..InstanceCreateInfo::application_from_cargo_toml()
        };

        let instance = Instance::new(library, app_info)
            .expect("Couldn't create instance");

        let cb = if enable_validation { 
            Some(Self::setup_debug_callback(&instance))
        } else {
            None
        };

        let surface = WindowBuilder::new()
            .with_title("Vulkan tutorial")
            .with_inner_size(logical_size)
            .build_vk_surface(event_loop, instance.clone())
            .expect("Couldn't create surface");

        let device_index = Self::pick_physical_device_index(&instance, &surface);

        Self {
            instance,
            surface,
            device_index,
        }
    }

    pub fn current_extent(&self) -> [u32; 2] {
        let capabilities = self.surface_capabilities();

        if let Some(cur_extent) = capabilities.current_extent {
            cur_extent
        } else {
            let max_0 = capabilities.max_image_extent[0].min(WIDTH);
            let max_1 = capabilities.max_image_extent[1].min(HEIGHT);

            let min_0 = capabilities.min_image_extent[0]; 
            let min_1 = capabilities.min_image_extent[1];

            [min_0.max(max_0), min_1.max(max_1)]
        }
    }

    pub fn surface_capabilities(&self) -> SurfaceCapabilities {
        self.physical_device()
            .surface_capabilities(&self.surface, SurfaceInfo::default())
            .expect("Can't read surface capabilities")
    }

    fn find_queue_families(device: &PhysicalDevice, surface: &Arc<Surface>) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices::new();

        for (i, props) in device.queue_family_properties().iter().enumerate() {
            if props.queue_flags.contains(QueueFlags::GRAPHICS) {
                indices.graphics_family = i as i32;
            }

            if device.surface_support(i as u32, surface).unwrap() {
                indices.present_family = i as i32;
            }

            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    pub fn get_queue_indices(&self) -> QueueFamilyIndices {
        let device = self.physical_device();
        Self::find_queue_families(&device, &self.surface)
    }

    fn check_device_extension_support(device: &PhysicalDevice) -> bool {
        let available_exts = device.supported_extensions();
        let required_exts = Self::device_extensions();
        available_exts.intersection(&required_exts) == required_exts
    }

    fn check_device_features_support(device: &PhysicalDevice) -> bool {
        let available_features = device.supported_features();
        let required_features = Self::device_features();
        available_features.intersection(&required_features) == required_features
    }

    fn is_device_suitable(device: &PhysicalDevice, surface: &Arc<Surface>) -> bool {
        let indices = Self::find_queue_families(device, surface);

        let extensions_supported = Self::check_device_extension_support(device);
        let features_supported = Self::check_device_features_support(device);

        let swap_chain_adequate = if extensions_supported {
            Self::check_swap_chain_adequate(device, surface)
        } else {
            false
        };

        indices.is_complete() && extensions_supported && features_supported && swap_chain_adequate
    }

    fn check_swap_chain_adequate(device: &PhysicalDevice, surface: &Arc<Surface>) -> bool {
        let info = SurfaceInfo::default();

        let formats = device
            .surface_formats(surface, info)
            .expect("Failed to get formats from the device");

        let present_modes = device
            .surface_present_modes(surface)
            .expect("Failed to get surface present modes");
  
        !formats.is_empty() && present_modes.into_iter().next().is_some()
    }

    fn pick_physical_device_index(instance: &Arc<Instance>, surface: &Arc<Surface>) -> usize {
        instance.enumerate_physical_devices()
            .expect("No physical device available")
            .into_iter()
            .position(|device| Self::is_device_suitable(&device, surface))
            .expect("failed to find suitable gpu")
    }

    pub fn physical_device(&self) -> Arc<PhysicalDevice> {
        self.instance.enumerate_physical_devices()
            .expect("No physical device available")
            .into_iter()
            .nth(self.device_index)
            .expect(&format!("No device with index {} found", self.device_index))
    }


    fn get_required_extensions(lib: &Arc<VulkanLibrary>, enable_validation: bool) -> InstanceExtensions {
        let mut extensions = vulkano_win::required_extensions(lib);
        extensions.ext_debug_utils = enable_validation;
        extensions
    }

    fn setup_debug_callback(instance: &Arc<Instance>) -> DebugUtilsMessenger {
        unsafe {
            DebugUtilsMessenger::new(
                instance.clone(),
                DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|msg| {
                    println!("Debug callback: {:?}", msg.description);
                })),
            ).expect("Can't setup debug callback")
        }
    }

    fn get_missing_validation_layers(lib: &Arc<VulkanLibrary>) -> Vec<String> {
        let available_layers: Vec<_> = lib.layer_properties()
            .unwrap()
            .map(|l| l.name().to_owned())
            .collect(); 
        
        VALIDATION_LAYERS
            .iter()
            .map(|l| l.to_string())
            .filter(|vl| !available_layers.contains(&vl))
            .collect()
    }

    fn ensure_validation_supported(lib: &Arc<VulkanLibrary>) {
        let missing_layers = Self::get_missing_validation_layers(lib);

        if !missing_layers.is_empty() {
            let msg = format!("Following layers are not supported: {}", missing_layers.join(", "));
            panic!("{}", msg);
        }
    }

    pub fn device_extensions() -> DeviceExtensions {
        DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        } 
    }

    pub fn device_features() -> Features {
        Features {
            sampler_anisotropy: true,
            ..Features::empty()
        }
    }
}
