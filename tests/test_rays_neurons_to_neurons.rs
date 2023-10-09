#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_rays_neurons_to_neurons() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let ZERO = arrayfire::constant::<f64>(0.0, single_dims);




    let mut input_pos_cpu:Vec<f64> = vec![ 0.1, -1.7, 0.3,     -0.1, 0.4, -0.7,          -0.3, -4.2, -3.9,       7.5, -0.7, -2.1,      -0.3,-0.6,0.9,            0.1,-0.7,0.5,         0.2, -0.7, -5.2  ];
    let mut input_pos = arrayfire::Array::new(&input_pos_cpu, arrayfire::Dim4::new(&[3, 7, 1, 1]));
    
    input_pos = arrayfire::transpose(&input_pos, false);
    



    let mut input_idx_cpu:Vec<i32> = vec![1, 2, 5,   7, 8, 9, 10  ];
    let mut input_idx = arrayfire::Array::new(&input_idx_cpu, arrayfire::Dim4::new(&[7, 1, 1, 1]));
    









    let mut hidden_pos_cpu:Vec<f64> = vec![ 2.6, -2.4, 4.3,     3.1, 1.4, -2.7,           -4.3, 1.2, -1.9,        2.5, 4.7, -1.1,       2.3,0.6,-0.9,      ];
    let mut hidden_pos = arrayfire::Array::new(&hidden_pos_cpu, arrayfire::Dim4::new(&[3, 5, 1, 1]));
    
    hidden_pos = arrayfire::transpose(&hidden_pos, false);
    

    let mut hidden_idx_cpu:Vec<i32> = vec![ 12, 15, 17, 18, 19  ];
    let mut hidden_idx = arrayfire::Array::new(&hidden_idx_cpu, arrayfire::Dim4::new(&[7, 1, 1, 1]));
    






    let mut hidden_size = hidden_pos.dims()[0];
    let input_idx_size = input_idx.dims()[0];



    //Generate rays starting from input neurons
    let mut start_line = ZERO.clone();
    let mut dir_line = ZERO.clone();

    

    let tile_dims = arrayfire::Dim4::new(&[hidden_size,1,1,1]);

    let mut tiled_input_idx =  arrayfire::tile(&input_idx, tile_dims);
    
    let mut tiled_hidden_idx = hidden_idx.clone();

    RayBNN_Raytrace::Generate::Fixed::tileDown(
        input_idx_size,
    
        &mut tiled_hidden_idx
    );

    RayBNN_Raytrace::Generate::Fixed::rays_from_neuronsA_to_neuronsB(
        con_rad,

        &input_pos,
        &hidden_pos,
    
        &mut start_line,
        &mut dir_line,

        &mut tiled_input_idx,
        &mut tiled_hidden_idx,
    );
    



}
