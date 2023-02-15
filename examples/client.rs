use image::ImageFormat;
use nalgebra_glm as glm;
use std::{
    borrow::Borrow,
    cell::RefCell,
    f32::consts::PI,
    io::Cursor,
    path::{Path, PathBuf},
    rc::Rc,
};

use glium::{
    glutin::{
        dpi::LogicalSize,
        event::{Event, WindowEvent},
        event_loop::EventLoop,
        window::WindowBuilder,
        ContextBuilder,
    },
    implement_vertex,
    index::{NoIndices, PrimitiveType},
    program, uniform, Depth, DepthTest, Display, DrawParameters, IndexBuffer, Surface,
    VertexBuffer,
};
use russimp::{
    material::{DataContent, Material, TextureType},
    node::Node,
    scene::{PostProcess, Scene},
};

enum Character {
    Aatrox,
}

impl Character {
    pub fn filepath(self) -> String {
        use Character::*;
        format!(
            "assets/{}.glb",
            match self {
                Aatrox => "88e5f2a34ef35bbd5fd9bc9aa647d047",
            }
        )
    }
}

#[derive(Debug)]
struct Model {
    meshes: Vec<Mesh>,
    filepath: PathBuf,
}

#[derive(Debug)]
struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    textures: Vec<glium::texture::SrgbTexture2d>,
}

#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    texture: [f32; 2],
}

implement_vertex!(Vertex, position, normal, texture);

struct Camera {
    view_matrix: [[f32; 4]; 4],
    perspective_matrix: [[f32; 4]; 4],
}

impl Camera {
    pub fn process_input(&self, event: &WindowEvent) {
        todo!()
    }
}

impl Model {
    pub fn load_character(display: &Display, character: Character) -> Self {
        use PostProcess::*;
        let filepath = character.filepath();
        let scene =
            Scene::from_file(filepath.as_str(), vec![Triangulate, FixOrRemoveInvalidData]).unwrap();

        let mut model = Self {
            meshes: Vec::new(),
            filepath: Path::new(&filepath).canonicalize().unwrap(),
        };

        model.process_node(scene.root.as_ref().unwrap(), &scene, display);

        model
    }

    fn process_node(&mut self, node: &RefCell<Node>, scene: &Scene, display: &Display) {
        // Process all the nodes meshes if any are present.
        node.borrow().meshes.iter().for_each(|mesh| {
            self.meshes.push(Self::process_mesh(
                scene.meshes.get(*mesh as usize).unwrap(),
                scene,
                display,
            ))
        });

        node.borrow().children.iter().for_each(|child| {
            self.process_node(child, scene, display);
        });
    }

    fn process_mesh(mesh: &russimp::mesh::Mesh, scene: &Scene, display: &Display) -> Mesh {
        let vertices: Vec<Vertex> = if let Some(textures) = mesh.texture_coords.first() {
            mesh.vertices
                .iter()
                .zip(mesh.normals.iter())
                .zip(textures.as_ref().unwrap().iter())
                .map(|((vertex, normal), texture)| Vertex {
                    position: [vertex.x, vertex.y, vertex.z],
                    normal: [normal.x, normal.y, normal.z],
                    texture: [texture.x, texture.y],
                })
                .collect()
        } else {
            mesh.vertices
                .iter()
                .zip(mesh.normals.iter())
                .map(|(vertex, normal)| Vertex {
                    position: [vertex.x, vertex.y, vertex.z],
                    normal: [normal.x, normal.y, normal.z],
                    texture: [0.0, 0.0],
                })
                .collect()
        };

        let indices = mesh.faces.iter().flat_map(|face| face.0.clone()).collect();

        let material = scene.materials.get(mesh.material_index as usize).unwrap();
        dbg!(&material.textures);
        let textures = [TextureType::Diffuse, TextureType::BaseColor]
            .iter()
            .map(|texture_type| {
                let texture = material.textures.get(texture_type).unwrap();
                let image = image::load_from_memory(match &texture.as_ref().borrow().data {
                    DataContent::Bytes(buffer) => buffer.as_slice(),
                    _ => unreachable!(),
                })
                .unwrap()
                .to_rgba8();
                let image_dimensions = image.dimensions();
                let image = glium::texture::RawImage2d::from_raw_rgba_reversed(
                    &image.into_raw(),
                    image_dimensions,
                );
                glium::texture::SrgbTexture2d::new(display, image).unwrap()
            })
            .collect();

        dbg!(&textures);

        Mesh {
            vertices,
            indices,
            textures,
        }
    }
}

fn main() {
    // The `winit::EventLoop` for handling events.
    let event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new().with_inner_size(LogicalSize::new(640.0, 480.0));
    let context_builder = ContextBuilder::new().with_depth_buffer(24);
    let display = Display::new(window_builder, context_builder, &event_loop).unwrap();

    let program = program!(&display,
        410 => {
            vertex: r#"
#version 410

uniform mat4 perspective_matrix;
uniform mat4 view_matrix;

in vec3 position;
in vec3 normal;
in vec2 texture_coordinates;

out vec3 vertex_position;
out vec3 vertex_normal;
out vec2 vertex_texture_coordinates;

void main() {
    vertex_texture_coordinates = texture_coordinates;
    vertex_normal = normal;
    gl_Position = perspective_matrix * view_matrix * vec4(vertex_position * 0.015, 1.0);
    vertex_position = gl_Position.xyz / gl_Position.w;
}
"#,

            fragment: r#"
#version 410

in vec3 vertex_normal;
in vec3 vertex_position;
in vec2 vertex_texture_coordinates;

uniform sampler2D diffuse_texture;

out vec4 fragment_color;

const vec3 LIGHT = vec3(-0.2, 0.8, 0.1);
const vec3 SPECULAR_COLOR = vec3(1.0, 1.0, 1.0);

void main() {
    vec3 diffuse_color = texture(diffuse_texture, vertex_texture_coordinates).rgb;
    vec3 ambient_color = diffuse_color * 0.1;

    float diffuse = max(dot(normalize(vertex_normal), normalize(LIGHT)), 0.0);

    vec3 camera_direction = normalize(-vertex_position);
    vec3 half_direction = normalize(normalize(LIGHT) + camera_direction);
    float specular = pow(max(dot(half_direction, normalize(vertex_normal)), 0.0), 16.0);

    fragment_color = vec4(ambient_color + diffuse * diffuse_color + specular * SPECULAR_COLOR, 1.0);
}
"#,
        },
    )
    .unwrap();

    let camera = Camera {
        view_matrix: glm::translate(&glm::identity(), &glm::vec3(-1.0, -1.0, -3.0)).into(),
        perspective_matrix: glm::perspective(PI / 2.0, 640.0 / 480.0, 0.1, 100.0).into(),
    };

    let model = Model::load_character(&display, Character::Aatrox);

    let mesh = model.meshes.leak().first().unwrap();

    let vertex_buffer = VertexBuffer::new(&display, &mesh.vertices).unwrap();
    let index_buffer =
        IndexBuffer::new(&display, PrimitiveType::TrianglesList, &mesh.indices).unwrap();

    dbg!(&mesh.textures);
    let diffuse_texture = mesh.textures.first().unwrap();

    event_loop.run(move |event, _, control_flow| {
        let next_frame_time =
            std::time::Instant::now() + std::time::Duration::from_nanos(16_666_667);
        control_flow.set_wait_until(next_frame_time);

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => control_flow.set_exit(),
                event => camera.process_input(&event),
            },
            _ => (),
        }

        let uniforms = uniform! {
            perspective_matrix: camera.perspective_matrix,
            view_matrix: camera.view_matrix,
            diffuse_texture: diffuse_texture,
        };

        let draw_parameters = DrawParameters {
            depth: Depth {
                test: DepthTest::IfLess,
                write: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut target = display.draw();

        target.clear_color_and_depth((0.0, 0.0, 0.0, 0.0), 1.0);
        target
            .draw(
                &vertex_buffer,
                &index_buffer,
                &program,
                &uniforms,
                &draw_parameters,
            )
            .unwrap();

        target.finish().unwrap();
    });
}
