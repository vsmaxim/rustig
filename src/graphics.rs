extern crate vulkano;
extern crate winit;
extern crate ahash;
extern crate vulkano_win;
extern crate vulkano_shaders;
extern crate nalgebra as na;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use image::io::Reader as ImageReader;
use image::DynamicImage;

use na::{Point3, Vector3};
use smallvec::smallvec;

use vulkano::DeviceSize;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents, CopyBufferInfo, PrimaryCommandBufferAbstract};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{DescriptorType, DescriptorSetLayout, DescriptorSetLayoutCreateInfo, DescriptorSetLayoutBinding};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, Queue};
use vulkano::format::{Format, ClearValue};
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{SwapchainImage, ImageUsage, SampleCount, ImageLayout, ImageSubresourceRange, ImageAspect, ImageAccess, ImmutableImage, ImageDimensions, MipmapsCount, AttachmentImage};
use vulkano::memory::allocator::{StandardMemoryAllocator, AllocationCreateInfo, MemoryUsage};
use vulkano::pipeline::graphics::color_blend::{ColorBlendState, ColorBlendAttachmentState, ColorComponents};
use vulkano::pipeline::graphics::depth_stencil::{DepthStencilState, DepthState, CompareOp};
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::pipeline::{GraphicsPipeline, StateMode, PartialStateMode, PipelineLayout, PipelineBindPoint, Pipeline};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::rasterization::{RasterizationState, PolygonMode, CullMode, FrontFace};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, AttachmentDescription, LoadOp, StoreOp, SubpassDescription, AttachmentReference, Subpass};
use vulkano::sampler::{Sampler, SamplerCreateInfo, Filter, SamplerAddressMode};
use vulkano::sync::{Sharing, self, GpuFuture, FlushError};
use vulkano::shader::{EntryPoint, ShaderStages};
use vulkano_win::VkSurfaceBuild;
use vulkano::swapchain::{Surface, SurfaceInfo, ColorSpace, PresentMode, SurfaceCapabilities, Swapchain, SwapchainCreateInfo, CompositeAlpha, acquire_next_image, SwapchainPresentInfo};

use vulkano::{instance::{
    Instance,
    InstanceExtensions,
}, VulkanLibrary, device::physical::PhysicalDevice};

use winit::{
    event_loop::EventLoop,
    window::WindowBuilder,
    dpi::LogicalSize
};

use crate::device::GameDevice;
use crate::object::RenderableObject;
use crate::ubo::UniformBufferObject;
use crate::vertex::MyVertex;
use crate::window::SceneView;

const WIDTH: u32 = 1200;
const HEIGHT: u32 = 1000;

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];
const ENABLE_VALIDATION_LAYERS: bool = true;
const MAX_FRAMES_IN_FLIGHT: usize = 2;


pub struct ApplicationView {
    game_device: Arc<GameDevice>,
    device: Arc<Device>,

    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,

    swap_chain: Arc<Swapchain>,
    swap_chain_images: Vec<Arc<SwapchainImage>>,
    swap_chain_outdated: bool,

    render_pass: Arc<RenderPass>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,

    swap_chain_framebuffers: Vec<Arc<Framebuffer>>,

    command_buffer_allocator: StandardCommandBufferAllocator,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: StandardDescriptorSetAllocator,

    index_device_buffers: Vec<Subbuffer<[u32]>>,
    index_staging_buffers: Vec<Subbuffer<[u32]>>,

    vertex_device_buffers: Vec<Subbuffer<[MyVertex]>>,
    vertex_staging_buffers: Vec<Subbuffer<[MyVertex]>>,
 
    to_load: Vec<bool>,

    uniform_buffers: SubbufferAllocator,

    current_frame: usize,
    previous_frame_end: Option<Box<dyn GpuFuture + 'static>>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    // command_buffers: Vec<Arc<dyn PrimaryCommandBufferAbstract>>,

    start_time: Instant,

    vertices: Vec<MyVertex>,
    indices: Vec<u32>,

    texture: DynamicImage,
    depth_fmt: Format,
    depth_image_view: Arc<ImageView<AttachmentImage>>,

    pub camera_position: Point3<f32>, 
    pub camera_look_at: Point3<f32>,
}

#[derive(Debug)]
enum ApplicationError {
    MissingVaildationLayers(String)
}

static VERTICES: [MyVertex; 8] = [
    // First rect
    MyVertex {
        pos: [-0.5, -0.5, 0.0],
        col: [1.0, 0.0, 0.0],
        tex_coord: [1.0, 0.0],
    },
    MyVertex {
        pos: [0.5, -0.5, 0.0],
        col: [0.0, 1.0, 0.0],
        tex_coord: [0.0, 0.0],
    },
    MyVertex {
        pos: [0.5, 0.5, 0.0],
        col: [0.0, 0.0, 1.0],
        tex_coord: [0.0, 1.0],
    },
    MyVertex {
        pos: [-0.5, 0.5, 0.0],
        col: [1.0, 1.0, 1.0],
        tex_coord: [1.0, 1.0],
    },

    // Second rect
    MyVertex {
        pos: [-0.5, -0.5, -0.5],
        col: [1.0, 0.0, 0.0],
        tex_coord: [1.0, 0.0],
    },
    MyVertex {
        pos: [0.5, -0.5, -0.5],
        col: [0.0, 1.0, 0.0],
        tex_coord: [0.0, 0.0],
    },
    MyVertex {
        pos: [0.5, 0.5, -0.5],
        col: [0.0, 0.0, 1.0],
        tex_coord: [0.0, 1.0],
    },
    MyVertex {
        pos: [-0.5, 0.5, -0.5],
        col: [1.0, 1.0, 1.0],
        tex_coord: [1.0, 1.0],
    },
];

// Indices definition
static INDICES: [u32; 12] = [
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4,
];


impl ApplicationView {
    pub fn new(game_device: &Arc<GameDevice>) -> Self {
        let start_time = Instant::now();
        let depth_fmt = Format::D32_SFLOAT;


        let (device, graphics_queue, present_queue) = Self::create_logical_device(&game_device);

        let (swap_chain, swap_chain_images) = Self::create_swap_chain(&game_device, &device);

        let memory_allocator = Self::create_memory_allocator(&device);
        let uniform_buffers = Self::create_uniform_buffer(&memory_allocator);
        let descriptor_set_allocator = Self::create_descriptor_set_allocator(&device);
        let descriptor_set_layout = Self::create_descriptor_set_layout(&device);

        let depth_image_view = Self::create_depth_resources(
            &memory_allocator,
            swap_chain.image_extent(),
            depth_fmt,
        );

        let render_pass = Self::create_render_pass(
            &device,
            swap_chain.image_format(),
            depth_fmt,
        );

        let graphics_pipeline = Self::create_graphics_pipeline(&device, &render_pass, &descriptor_set_layout);
        let viewport = Self::create_viewport(&swap_chain);

        let swap_chain_framebuffers = Self::create_framebuffers(
            &swap_chain_images,
            &render_pass,
            swap_chain.image_format(),
            &depth_image_view,
        );

        let command_buffer_allocator = Self::create_cb_allocator(&device);
        let current_frame = 0;

        let (vertices, indices) = Self::load_object();
        let (index_staging_buffer, index_device_buffer) = Self::create_index_buffers(&memory_allocator, &indices);
        let (vertex_staging_buffer, vertex_device_buffer) = Self::create_vertex_bufers(&memory_allocator, &vertices);

        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        let texture = Self::load_texture();

        let camera_position = Point3::new(2.0, 2.0, 2.0);
        let camera_look_at = Point3::new(0.0, 0.0, 0.0);

        Self { 
            start_time,
            game_device: game_device.clone(),
            device,
            graphics_queue,
            present_queue,

            swap_chain,
            swap_chain_images,
            swap_chain_outdated: false,

            render_pass,
            graphics_pipeline,
            viewport,

            swap_chain_framebuffers,

            command_buffer_allocator,
            memory_allocator,
            descriptor_set_allocator,

            index_staging_buffers: Vec::new(),
            index_device_buffers: Vec::new(),
            vertex_staging_buffers: Vec::new(),
            vertex_device_buffers: Vec::new(),
            to_load: Vec::new(),

            uniform_buffers,

            current_frame,
            previous_frame_end,

            descriptor_set_layout,

            texture,

            depth_fmt,
            depth_image_view,

            vertices,
            indices,

            camera_position,
            camera_look_at,
        }
    }

    fn load_texture() -> DynamicImage {
        ImageReader::open("src/textures/viking_room.png").unwrap().decode().unwrap()
    }

    fn load_object() -> (Vec<MyVertex>, Vec<u32>) {
        let (models, _) = tobj::load_obj(
            "src/models/viking_room.obj",
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

    fn create_depth_resources(
        allocator: &StandardMemoryAllocator,
        swapchain_extent: [u32; 2],
        depth_fmt: Format, 
    ) -> Arc<ImageView<AttachmentImage>> {
        let depth_img = AttachmentImage::new(
            allocator,
            swapchain_extent,
            depth_fmt,
        ).unwrap();

        ImageView::new(depth_img.clone(), ImageViewCreateInfo {
            format: Some(depth_fmt), 
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspect::Depth.into(),
                mip_levels: 0..depth_img.mip_levels(),
                array_layers: 0..depth_img.dimensions().array_layers(),
            },
            ..ImageViewCreateInfo::default()
        }).unwrap()
    }

    fn create_descriptor_sets(
        allocator: &StandardDescriptorSetAllocator,
        layout: &Arc<DescriptorSetLayout>,
        buffer: &Subbuffer<UniformBufferObject>,
        sampler: &Arc<Sampler>,
        view: &Arc<ImageView<ImmutableImage>>,
    ) -> Arc<PersistentDescriptorSet> {
        PersistentDescriptorSet::new(
            allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer.clone()),
                WriteDescriptorSet::image_view_sampler(1, view.clone(), sampler.clone()),
            ],
        ).unwrap()
    }

    fn create_descriptor_set_layout(device: &Arc<Device>) -> Arc<DescriptorSetLayout> {
        DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo { 
                bindings: [
                    (0, DescriptorSetLayoutBinding {
                        descriptor_count: 1,
                        stages: ShaderStages::VERTEX,
                        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
                    }),
                    (1, DescriptorSetLayoutBinding {
                        descriptor_count: 1,
                        stages: ShaderStages::FRAGMENT,
                        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler)
                    })
                ].into(),
                push_descriptor: false,
                ..Default::default()
            }
        ).expect("Can't create descriptor set layout")
    }

    fn create_descriptor_set_allocator(device: &Arc<Device>) -> StandardDescriptorSetAllocator {
        StandardDescriptorSetAllocator::new(device.clone())
    }

    fn create_memory_allocator(device: &Arc<Device>) -> Arc<StandardMemoryAllocator> {
        Arc::new(StandardMemoryAllocator::new_default(device.clone()))
    }

    fn create_uniform_buffer(memory_allocator: &Arc<StandardMemoryAllocator>) -> SubbufferAllocator {
        SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_usage: MemoryUsage::Upload,
                ..Default::default()
            }
        )
    
    }

    fn next_uniform_buffer(&mut self) -> UniformBufferObject {  
        let [swap_w, swap_h] = self.swap_chain.image_extent();
        let aspect = swap_w as f32 / swap_h as f32;
        UniformBufferObject::new(aspect, &self.camera_position)
    }

    fn load_to_gpu(&mut self) {
        self.to_load = self.to_load
            .iter().enumerate()
            .map(|(buf_ind, should_load)| {
                if *should_load {
                    println!("Loading buffer to gpu");

                    let mut load_cbb = AutoCommandBufferBuilder::primary(
                        &self.command_buffer_allocator,
                        self.graphics_queue.queue_family_index(),
                        CommandBufferUsage::OneTimeSubmit,
                    ).unwrap();

                    load_cbb
                        .copy_buffer(CopyBufferInfo::buffers(
                            self.vertex_staging_buffers[buf_ind].clone(),
                            self.vertex_device_buffers[buf_ind].clone(),
                        )).unwrap()
                        .copy_buffer(CopyBufferInfo::buffers(
                            self.index_staging_buffers[buf_ind].clone(),
                            self.index_device_buffers[buf_ind].clone(),
                        )).unwrap();

                    let load_cb = load_cbb.build().unwrap(); 

                    load_cb
                        .execute(self.graphics_queue.clone()).unwrap()
                        .then_signal_fence_and_flush().unwrap()
                        .wait(None)
                        .unwrap();
                }

                false
            }).collect();

    }

    pub fn render_object(
        &mut self,
        o: RenderableObject,
        move_delta: &Vector3<f32>,
    ) { 
        let vertices = o.vertices
            .iter()
            .map(|v| MyVertex {
                pos: (Point3::from(v.pos) + move_delta).into(),
                col: v.col,
                tex_coord: v.tex_coord,
            })
            .collect();

        let (vertex_staging_buf, vertex_device_buf) = Self::create_vertex_bufers(&self.memory_allocator, &vertices);
        let (index_staging_buf, index_device_buf) = Self::create_index_buffers(&self.memory_allocator, &o.indices);

        self.vertex_staging_buffers.push(vertex_staging_buf);
        self.vertex_device_buffers.push(vertex_device_buf);
        self.index_staging_buffers.push(index_staging_buf);
        self.index_device_buffers.push(index_device_buf);
        self.to_load.push(true);
    }

    fn create_vertex_bufers(
        memory_allocator: &StandardMemoryAllocator,
        vertices: &Vec<MyVertex>,
    ) -> (Subbuffer<[MyVertex]>, Subbuffer<[MyVertex]>) {
        let vertex_staging_buffer = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo { 
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vertices.clone().into_iter(),
        ).unwrap();

        let vertex_device_buffer = Buffer::new_slice::<MyVertex>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST | BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo { 
                usage: MemoryUsage::DeviceOnly,
                ..Default::default()
            },  
            vertices.len().try_into().unwrap(),
        ).unwrap();

        (vertex_staging_buffer, vertex_device_buffer)
    }

    fn create_index_buffers(
        memory_allocator: &StandardMemoryAllocator,
        indices: &Vec<u32>,
    ) -> (Subbuffer<[u32]>, Subbuffer<[u32]>) {
        let index_staging_buffer = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo { 
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo { 
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            indices.iter().copied(),
        ).unwrap();

        let index_device_buffer = Buffer::new_slice::<u32>(
            memory_allocator,
            BufferCreateInfo { 
                usage: BufferUsage::TRANSFER_DST | BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo { 
                usage: MemoryUsage::DeviceOnly,
                ..Default::default()
            },
            indices.len() as DeviceSize,
        ).unwrap();

        (index_staging_buffer, index_device_buffer)
    }

    fn create_framebuffers(
        swap_chain_images: &Vec<Arc<SwapchainImage>>,
        render_pass: &Arc<RenderPass>,
        format: Format,
        depth_image_view: &Arc<ImageView<AttachmentImage>>,
    ) -> Vec<Arc<Framebuffer>> {
        swap_chain_images
            .iter()
            .map(|image| {
                let view = ImageView::new(
                    image.clone(),
                    ImageViewCreateInfo {
                        format: Some(format), 
                        subresource_range: ImageSubresourceRange {
                            aspects: ImageAspect::Color.into(),
                            mip_levels: 0..image.mip_levels(),
                            array_layers: 0..image.dimensions().array_layers(),
                        },
                        ..ImageViewCreateInfo::default()
                    }
                )
                    .expect("failed to create image view");

                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo { 
                        attachments: vec![
                            view,
                            depth_image_view.clone(),
                        ],
                        ..Default::default()
                    }
                ).expect("failed to create framebuffers")
            })
            .collect()
    }

    fn create_cb_allocator(device: &Arc<Device>) -> StandardCommandBufferAllocator {
        StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        )
    }

    fn create_sync_primitives(device: &Arc<Device>, count: usize) -> Vec<Option<Box<dyn GpuFuture + 'static>>> {
        (0..count)
            .map(|_| Some(sync::now(device.clone()).boxed()))
            .collect()
    }


    fn create_render_pass(
        device: &Arc<Device>,
        color_format: Format,
        depth_format: Format,
    ) -> Arc<RenderPass> {
        RenderPass::new(  
            device.clone(),
            RenderPassCreateInfo { 
                attachments: vec![
                    AttachmentDescription {
                        format: color_format.into(),
                        samples: SampleCount::Sample1, 
                        load_op: LoadOp::Clear,
                        store_op: StoreOp::Store,
                        stencil_load_op: LoadOp::DontCare,
                        stencil_store_op: StoreOp::DontCare,
                        initial_layout: ImageLayout::Undefined,
                        final_layout: ImageLayout::PresentSrc,
                        ..Default::default()
                    },
                    AttachmentDescription {
                        format: depth_format.into(),
                        samples: SampleCount::Sample1,
                        load_op: LoadOp::Clear,
                        store_op: StoreOp::DontCare,
                        stencil_load_op: LoadOp::DontCare,
                        stencil_store_op: StoreOp::DontCare,
                        initial_layout: ImageLayout::Undefined,
                        final_layout: ImageLayout::DepthStencilAttachmentOptimal,
                        ..Default::default()
                    },
                ],
                subpasses: vec![
                    SubpassDescription {
                        color_attachments: vec![
                            AttachmentReference {
                                attachment: 0,
                                layout: ImageLayout::ColorAttachmentOptimal,
                                ..Default::default()
                            }.into()
                        ],
                        depth_stencil_attachment: AttachmentReference {
                            attachment: 1,
                            layout: ImageLayout::DepthStencilAttachmentOptimal,
                            ..Default::default()
                        }.into(),
                        ..Default::default()
                    }
                ],
                ..Default::default()
            }
        ).expect("Couldn't create render pass")
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        render_pass: &Arc<RenderPass>,
        descriptor_set_layout: &Arc<DescriptorSetLayout>,
    ) -> Arc<GraphicsPipeline> {
        mod vertex_shader {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "src/shaders/shader.vert"
            }
        }

        mod fragment_shader {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "src/shaders/shader.frag"
            }
        }

        let vs = vertex_shader::load(device.clone()).expect("failed to load vertex shader");
        let fs = fragment_shader::load(device.clone()).expect("Failed to load frag shader");

        let vertex_entrypoint: EntryPoint = vs.entry_point("main")
            .expect("Failed to load vertex entrypoint 'main'");

        let fragment_entrypoint: EntryPoint = fs.entry_point("main")
            .expect("Failed to load fragment entrypoint 'main'");

        let input_state = MyVertex::per_vertex()
            .definition(&vertex_entrypoint.input_interface())
            .expect("Failed to construct input state");

        let input_assembly_state = InputAssemblyState {
            topology: PartialStateMode::Fixed(PrimitiveTopology::TriangleList),
            primitive_restart_enable: StateMode::Fixed(false),
        };

        let viewport_state = ViewportState::viewport_dynamic_scissor_irrelevant();

        let rasterization_state = RasterizationState {
            depth_clamp_enable: false,
            polygon_mode: PolygonMode::Fill,
            line_width: StateMode::Fixed(1.0),
            cull_mode: StateMode::Fixed(CullMode::Back),
            front_face: StateMode::Fixed(FrontFace::Clockwise),
            ..RasterizationState::default()
        };

        let color_blend_state = ColorBlendState {
            attachments: vec![
                ColorBlendAttachmentState {
                    blend: None,
                    color_write_mask: ColorComponents::all(),
                    color_write_enable: StateMode::Fixed(true),
                }
            ],
            blend_constants: StateMode::Fixed([0.0, 0.0, 0.0, 0.0]),
            ..Default::default()
        };

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![descriptor_set_layout.clone()],
                ..Default::default()
            },
        ).unwrap();

        let depth_stencil_state = DepthStencilState {
            depth: DepthState {
                enable_dynamic: false,
                write_enable: StateMode::Fixed(true),
                compare_op: StateMode::Fixed(CompareOp::Less),
            }.into(),
            stencil: None,
            depth_bounds: None,
        };

        let pipeline = GraphicsPipeline::start()
            .depth_stencil_state(depth_stencil_state)
            .vertex_input_state(input_state)
            .vertex_shader(vertex_entrypoint, ())
            .input_assembly_state(input_assembly_state)
            .viewport_state(viewport_state)
            .fragment_shader(fragment_entrypoint, ())
            .rasterization_state(rasterization_state)
            .color_blend_state(color_blend_state)
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .with_pipeline_layout(device.clone(), layout.clone()).unwrap();

        pipeline
    }

    fn create_surface(instance: &Arc<Instance>) -> (EventLoop<()>, Arc<Surface>) {
        let event_loop = EventLoop::new();
        let logical_size = LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT));
        let surface = WindowBuilder::new()
            .with_title("Vulkan tutorial")
            .with_inner_size(logical_size)
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();

        (event_loop, surface)
    }

    fn get_required_extensions(lib: &Arc<VulkanLibrary>) -> InstanceExtensions {
        let mut extensions = vulkano_win::required_extensions(lib);
        extensions.ext_debug_utils = ENABLE_VALIDATION_LAYERS;
        extensions
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

    fn ensure_validation_supported(lib: &Arc<VulkanLibrary>) -> Result<(), ApplicationError> {
        let missing_layers = Self::get_missing_validation_layers(lib);

        if missing_layers.is_empty() {
            Ok(())
        } else {
            let msg = format!("Following layers are not supported: {}", missing_layers.join(", "));
            Err(ApplicationError::MissingVaildationLayers(msg))
        }
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
    
    fn pick_swap_surface_format(available_fmts: Vec<(Format, ColorSpace)>) -> (Format, ColorSpace) {
        *available_fmts
            .iter()
            .find(|(fmt, cs)| *fmt == Format::B8G8R8A8_UNORM && *cs == ColorSpace::SrgbNonLinear)
            .unwrap_or_else(|| &available_fmts[0])
    }


    fn pick_swap_present_mode(available_modes: Vec<PresentMode>) -> PresentMode {
        let preferred_modes = vec![
            PresentMode::Mailbox,
            PresentMode::Immediate,
            PresentMode::Fifo,
        ];

        // Pick first mode from preferred in that order
        *preferred_modes
            .iter()
            .filter(|pm| available_modes.contains(&pm))
            .next()
            .expect("No present mode is available from preferred modes")
    }

    fn pick_swap_extent(capabilities: &SurfaceCapabilities) -> [u32; 2] {
        if let Some(current_extent) = capabilities.current_extent {
            current_extent
        } else {
            let max_0 = capabilities.max_image_extent[0].min(WIDTH);
            let max_1 = capabilities.max_image_extent[1].min(HEIGHT);

            [capabilities.min_image_extent[0].max(max_0), capabilities.min_image_extent[1].max(max_1)]
        }
    }

    fn get_swap_extent(device: &Arc<PhysicalDevice>, surface: &Arc<Surface>) -> [u32; 2] {
        let capabilities = device
            .surface_capabilities(surface, SurfaceInfo::default())
            .expect("Can't read surface capabilities"); 

        Self::pick_swap_extent(&capabilities)
    }

    fn recreate_swap_chain(&mut self, image_extent: [u32; 2]) {
        (self.swap_chain, self.swap_chain_images) = self.swap_chain.recreate(
            SwapchainCreateInfo { 
                image_extent,
                ..self.swap_chain.create_info()
            }
        ).unwrap();

        self.depth_image_view = Self::create_depth_resources(
            &self.memory_allocator,
            image_extent,
            self.depth_fmt,
        );

        self.swap_chain_framebuffers = Self::create_framebuffers(
            &self.swap_chain_images,
            &self.render_pass,
            self.swap_chain.image_format(),
            &self.depth_image_view,
        );

        self.viewport = Self::create_viewport(&self.swap_chain);

    }

    fn create_swap_chain(
        game_device: &GameDevice,
        device: &Arc<Device>,
    ) -> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) {    
        let physical_device = game_device.physical_device();
        let capabilities = game_device.surface_capabilities();

        let surface_info = SurfaceInfo::default();
        let surface_formats = physical_device.surface_formats(&game_device.surface, surface_info).unwrap();
        let surface_format = Self::pick_swap_surface_format(surface_formats);

        let present_modes = physical_device.surface_present_modes(&game_device.surface).unwrap().collect();
        let present_mode = Self::pick_swap_present_mode(present_modes); 

        let swap_extent = Self::pick_swap_extent(&capabilities);
        let indices = game_device.get_queue_indices();

        let image_sharing = if indices.graphics_family != indices.present_family {
            Sharing::Concurrent(
                smallvec![
                    indices.graphics_family as u32,
                    indices.present_family as u32,
                ]
            )
        } else {
            Sharing::Exclusive
        };

        let min_image_count = capabilities.min_image_count + 1;
        let max_image_count = capabilities.max_image_count.unwrap_or(min_image_count);


        let (swapchain, images) = Swapchain::new(
            device.clone(),
            game_device.surface.clone(),
            SwapchainCreateInfo {  
                min_image_count: min_image_count.min(max_image_count),
                image_format: Some(surface_format.0),
                image_color_space: surface_format.1,
                image_extent: swap_extent,
                image_array_layers: 1,   
                image_usage:  ImageUsage::COLOR_ATTACHMENT,
                image_sharing,
                pre_transform: capabilities.current_transform,
                present_mode,
                composite_alpha: CompositeAlpha::Opaque,
                clipped: true,
                ..SwapchainCreateInfo::default()
            },
        ).expect("Failed to create swapchain");

        (swapchain, images)
    }


    fn create_logical_device(device: &GameDevice) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {

        let physical_device = device.physical_device();
        let indices = device.get_queue_indices();

        let families = [indices.graphics_family, indices.present_family];
        let unique_families: HashSet<&i32> = HashSet::from_iter(families.iter());

        let queue_priority = 1.0;
        let queue_infos = unique_families
            .iter()
            .map(|&&family_index| QueueCreateInfo {
                queue_family_index: family_index as u32,
                queues: vec![queue_priority],
                ..Default::default()
            })
            .collect();

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo { 
                queue_create_infos: queue_infos,
                enabled_extensions: GameDevice::device_extensions(), 
                enabled_features: GameDevice::device_features(),
                ..Default::default()
            }
        ).expect("Can't create logical device");

        let graphics_queue = queues.next().unwrap();
        let present_queue = queues.next()
            .unwrap_or_else(|| graphics_queue.clone());        

        (device, graphics_queue, present_queue)
    }

    fn create_viewport(swapchain: &Arc<Swapchain>) -> Viewport {
        let extent = swapchain.image_extent();

        return Viewport { 
            origin: [0.0, 0.0],
            dimensions: [extent[0] as f32, extent[1] as f32],
            depth_range: 0.0..1.0
        };
    } 
}

impl SceneView for ApplicationView {
    fn window_resized(&mut self) {  
        self.swap_chain_outdated = true;
    }

    fn draw_frame(&mut self, extent: [u32; 2]) {
        if extent.contains(&0) { return; }

        if self.swap_chain_outdated {
            self.recreate_swap_chain(extent);
        }

        self.previous_frame_end.as_mut().unwrap().cleanup_finished();
        self.load_to_gpu();

        let (image_index, suboptimal, acquire_future) =  acquire_next_image(self.swap_chain.clone(), None).unwrap();
        let current_frame = image_index as usize;

        if suboptimal {
            self.swap_chain_outdated = true;
        }

        let uniform_buffer_subbuffer = {
            let subbuffer = self.uniform_buffers.allocate_sized().unwrap();
            *subbuffer.write().unwrap() = self.next_uniform_buffer();
            subbuffer
        };

        let cmd_buffers = (0..self.vertex_device_buffers.len())
            .into_iter()
            .map(|buf_ind| {
                let game_device = &self.game_device;
                let index_device_buffers = &mut self.index_device_buffers;

                let mut buffer_builder = AutoCommandBufferBuilder::primary(
                    &self.command_buffer_allocator,
                    self.graphics_queue.queue_family_index(),
                    CommandBufferUsage::MultipleSubmit,
                ).unwrap();

                let imm_img = ImmutableImage::from_iter(
                    &self.memory_allocator,
                    self.texture.clone().into_rgba8().iter().map(|v| *v),
                    ImageDimensions::Dim2d { 
                        width: self.texture.width(), 
                        height: self.texture.height(),
                        array_layers: 1u32,
                    },
                    MipmapsCount::One,
                    Format::R8G8B8A8_SRGB,
                    &mut buffer_builder,
                ).unwrap();

                let imm_view = ImageView::new(imm_img.clone(), ImageViewCreateInfo {
                    view_type: ImageViewType::Dim2d,
                    format: Some(Format::R8G8B8A8_SRGB),
                    subresource_range: ImageSubresourceRange {
                        aspects: ImageAspect::Color.into(),
                        mip_levels: 0..imm_img.mip_levels(),
                        array_layers: 0..imm_img.dimensions().array_layers(),
                    },
                    ..Default::default()
                }).unwrap();

                let sampler = Sampler::new(self.device.clone(), SamplerCreateInfo {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    address_mode: [SamplerAddressMode::Repeat; 3],
                    lod: 0.0..=100.0,
                    anisotropy: Some(
                        game_device
                            .physical_device()
                            .properties()
                            .max_sampler_anisotropy
                    ),
                    ..Default::default()
                }).unwrap();

                let descriptor_set = Self::create_descriptor_sets(
                    &self.descriptor_set_allocator,
                    self.graphics_pipeline.layout().set_layouts().get(0).unwrap(),
                    &uniform_buffer_subbuffer,
                    &sampler,
                    &imm_view,
                );

                buffer_builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![
                                // Should be identical to order of attachments
                                ClearValue::Float([0.0, 0.0, 0.0, 0.0]).into(),
                                ClearValue::Depth(1.0f32).into(),
                            ],
                            render_area_extent: self.swap_chain.image_extent(),
                            render_pass: self.render_pass.clone(), 
                            ..RenderPassBeginInfo::framebuffer(self.swap_chain_framebuffers[current_frame].clone())
                        },
                        SubpassContents::Inline,
                    ).unwrap() 
                    .bind_pipeline_graphics(self.graphics_pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        self.graphics_pipeline.layout().clone(),
                        0,
                        descriptor_set.clone(),
                    )
                    .bind_index_buffer(self.index_device_buffers[buf_ind].clone())
                    .bind_vertex_buffers(0, self.vertex_device_buffers[buf_ind].clone())
                    .set_viewport(0, [self.viewport.clone()].into_iter())
                    .draw_indexed(self.indices.len() as u32, 1, 0, 0, 0).unwrap()
                    .end_render_pass().unwrap();

                buffer_builder.build().unwrap()
        });
  
        acquire_future.wait(None).unwrap();

        let previous_frame_end = self.previous_frame_end.take()
            .unwrap()
            .join(acquire_future)
            .boxed();

        let new_frame_end = cmd_buffers
            .fold(previous_frame_end, |prev_frame, cmd_buf| {  
                let graphics_queue = self.graphics_queue.clone();
                prev_frame
                    .then_execute(graphics_queue.clone(), cmd_buf).unwrap()
                    .boxed()
            })
            .then_swapchain_present(
                self.graphics_queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swap_chain.clone(), image_index)
            )
            .then_signal_fence_and_flush();

        self.previous_frame_end = match new_frame_end {
            Ok(future) => {
                Some(future.boxed())
            }
            Err(FlushError::OutOfDate) => {
                self.swap_chain_outdated = true;
                Some(sync::now(self.device.clone()).boxed())
            }
            Err(e) => {
                panic!("failed to flush future: {e}");
            }
        }
    }

    fn init_view(&mut self) {
    }
}
 
