use nalgebra_glm as glm;
use std::{
    cell::RefCell,
    f32::consts::PI,
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
    program, uniform, Depth, DepthTest, Display, DrawParameters, Surface, VertexBuffer,
};
use russimp::{
    mesh::Mesh,
    node::Node,
    scene::{PostProcess, Scene},
};

#[derive(Debug)]
struct LocalModel {
    meshes: Vec<LocalMesh>,
    filepath: PathBuf,
}

#[derive(Debug)]
struct LocalMesh {
    vertices: Vec<LocalVertex>,
    indices: Vec<u32>,
    textures: Vec<LocalTexture>,
}

#[derive(Debug)]
struct LocalTexture {}

#[derive(Copy, Clone, Debug)]
struct LocalVertex {
    position: [f32; 3],
    normal: [f32; 3],
    texture: [f32; 2],
}

implement_vertex!(LocalVertex, position, normal, texture);

struct Camera {
    view_matrix: [[f32; 4]; 4],
    perspective_matrix: [[f32; 4]; 4],
}

impl LocalModel {
    pub fn load_from_filepath(filepath: &'static str) -> Self {
        let scene = Scene::from_file(
            filepath,
            vec![PostProcess::Triangulate, PostProcess::FlipUVs],
        )
        .unwrap();

        let mut model = Self {
            meshes: Vec::new(),
            filepath: Path::new(&filepath).canonicalize().unwrap(),
        };

        model.process_node(scene.root.as_ref().unwrap(), &scene);

        model
    }

    fn process_node(&mut self, node: &RefCell<Node>, scene: &Scene) {
        // Process all the nodes meshes if any are present.
        node.borrow().meshes.iter().for_each(|mesh| {
            self.meshes.push(Self::process_mesh(
                scene.meshes.get(*mesh as usize).unwrap(),
                scene,
            ))
        });

        node.borrow().children.iter().for_each(|child| {
            self.process_node(child, scene);
        });
    }

    fn process_mesh(mesh: &Mesh, scene: &Scene) -> LocalMesh {
        let vertices: Vec<LocalVertex> = if let Some(textures) = mesh.texture_coords.first() {
            mesh.vertices
                .iter()
                .zip(mesh.normals.iter())
                .zip(textures.as_ref().unwrap().iter())
                .map(|((vertex, normal), texture)| LocalVertex {
                    position: [vertex.x, vertex.y, vertex.z],
                    normal: [normal.x, normal.y, normal.z],
                    texture: [texture.x, texture.y],
                })
                .collect()
        } else {
            mesh.vertices
                .iter()
                .zip(mesh.normals.iter())
                .map(|(vertex, normal)| LocalVertex {
                    position: [vertex.x, vertex.y, vertex.z],
                    normal: [normal.x, normal.y, normal.z],
                    texture: [0.0, 0.0],
                })
                .collect()
        };

        let indices = mesh.faces.iter().flat_map(|face| face.0.clone()).collect();

        LocalMesh {
            vertices,
            indices,
            textures: Vec::new(),
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

out vec3 vertex_position;
out vec3 vertex_normal;

void main() {
    vertex_position = position;
    vertex_normal = normal;
    gl_Position = perspective_matrix * view_matrix * vec4(vertex_position * 0.015, 1.0);
}
"#,

            fragment: r#"
#version 410

in vec3 vertex_normal;

out vec4 fragment_color;

const vec3 LIGHT = vec3(-0.2, 0.8, 0.1);

void main() {
    float luminosity = max(dot(normalize(vertex_normal), normalize(LIGHT)), 0.0);
    vec3 color = (0.3 + 0.7 * luminosity) * vec3(1.0, 1.0, 1.0);
    fragment_color = vec4(color, 1.0);
}
"#,
        },
    )
    .unwrap();

    let camera = Camera {
        view_matrix: glm::translate(&glm::identity(), &glm::vec3(-1.0, -1.0, -3.0)).into(),
        perspective_matrix: glm::perspective(PI / 2.0, 640.0 / 480.0, 0.1, 100.0).into(),
    };

    let model = LocalModel::load_from_filepath("assets/88e5f2a34ef35bbd5fd9bc9aa647d047.glb");
    let vertex_buffer =
        VertexBuffer::new(&display, &model.meshes.first().unwrap().vertices).unwrap();

    event_loop.run(move |event, _, control_flow| {
        let uniforms = uniform! {
            perspective_matrix: camera.perspective_matrix,
            view_matrix: camera.view_matrix,
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
                &NoIndices(PrimitiveType::TrianglesList),
                &program,
                &uniforms,
                &draw_parameters,
            )
            .unwrap();

        target.finish().unwrap();

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => control_flow.set_exit(),
                event => (), // camera.process_input(&event),
            },
            _ => (),
        }
    });
}
