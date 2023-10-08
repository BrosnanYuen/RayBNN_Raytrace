#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_random_rays2() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


	arrayfire::set_seed(1231);








    let con_rad: f64 = 0.6;
    let space_dims: u64 = 2;


    let mut neuron_pos_cpu:Vec<f64> = vec![ 0.1, -1.7,      -0.1, 0.4,          -0.3, -4.2,        7.5, -0.7,      -0.3,-0.6,           0.1,-0.7,        0.2, -0.7 ];
    let mut neuron_pos = arrayfire::Array::new(&neuron_pos_cpu, arrayfire::Dim4::new(&[2, 7, 1, 1]));

    neuron_pos = arrayfire::transpose(&neuron_pos, false);

    let ray_num = 46;

    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

    let mut start_line = arrayfire::constant::<f64>(0.0,single_dims);
    let mut dir_line = arrayfire::constant::<f64>(0.0,single_dims);



    RayBNN_Raytrace::Generate::Random::generate_random_uniform_rays(
            &neuron_pos,
            ray_num,
            con_rad,
        
            &mut start_line,
            &mut dir_line
        );

    assert_eq!(start_line.dims()[0], neuron_pos.dims()[0]*ray_num );
    assert_eq!(start_line.dims()[1], space_dims );


    assert_eq!(dir_line.dims()[0], neuron_pos.dims()[0]*ray_num );
    assert_eq!(dir_line.dims()[1], space_dims );







    let tile_dims = arrayfire::Dim4::new(&[ray_num,1,1,1]);

    let start_line_act =  arrayfire::tile(&neuron_pos, tile_dims);

    let mut start_line_act_cpu = vec!(f64::default();start_line_act.elements());
    start_line_act.host(&mut start_line_act_cpu);

    let mut start_line_pred_cpu = vec!(f64::default();start_line.elements());
    start_line.host(&mut start_line_pred_cpu);

    start_line_act_cpu = start_line_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    start_line_pred_cpu = start_line_pred_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();

    assert_eq!(start_line_act_cpu, start_line_pred_cpu);





    let two = 2.0;


    let mut magsq = arrayfire::pow(&dir_line,&two,false);
    magsq = arrayfire::sum(&magsq, 1);
    magsq = arrayfire::sqrt(&magsq);



    let (std,_) = arrayfire::stdev_all_v2(&magsq,arrayfire::VarianceBias::SAMPLE);

    assert_eq!((std * 1000000.0).round() / 1000000.0  ,  0.0);






    let ( mut sumdist,_) = arrayfire::sum_all::<f64>(&magsq);

    let mut sumact: f64 = con_rad*(start_line.dims()[0] as f64);

    sumact = (sumact * 100000000.0).round() / 100000000.0 ;

    sumdist = (sumdist * 100000000.0).round() / 100000000.0 ;


    assert_eq!(sumdist, sumact);

























    let con_rad: f64 = 0.6;
    let space_dims: u64 = 3;
    
    
    let mut neuron_pos_cpu:Vec<f64> = vec![ 0.1, -1.7, 0.3,     -0.1, 0.4, -0.7,          -0.3, -4.2, -3.9,       7.5, -0.7, -2.1,      -0.3,-0.6,0.9,            0.1,-0.7,0.5,         0.2, -0.7, -5.2  ];
    let mut neuron_pos = arrayfire::Array::new(&neuron_pos_cpu, arrayfire::Dim4::new(&[3, 7, 1, 1]));
    
    neuron_pos = arrayfire::transpose(&neuron_pos, false);
    
    let ray_num = 23;
    
    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    
    let mut start_line = arrayfire::constant::<f64>(0.0,single_dims);
    let mut dir_line = arrayfire::constant::<f64>(0.0,single_dims);
    
    RayBNN_Raytrace::Generate::Random::generate_random_uniform_rays(
        &neuron_pos,
        ray_num,
        con_rad,
    
        &mut start_line,
        &mut dir_line
        );
    
    assert_eq!(start_line.dims()[0], neuron_pos.dims()[0]*ray_num );
    assert_eq!(start_line.dims()[1], space_dims );
    
    
    assert_eq!(dir_line.dims()[0], neuron_pos.dims()[0]*ray_num );
    assert_eq!(dir_line.dims()[1], space_dims );
    
    
    
    
    
    
    
    let tile_dims = arrayfire::Dim4::new(&[ray_num,1,1,1]);
    
    let start_line_act =  arrayfire::tile(&neuron_pos, tile_dims);
    
    let mut start_line_act_cpu = vec!(f64::default();start_line_act.elements());
    start_line_act.host(&mut start_line_act_cpu);
    
    let mut start_line_pred_cpu = vec!(f64::default();start_line.elements());
    start_line.host(&mut start_line_pred_cpu);
    
    start_line_act_cpu = start_line_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();
    
    start_line_pred_cpu = start_line_pred_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();
    
    assert_eq!(start_line_act_cpu, start_line_pred_cpu);
    
    
    
    
    
    
    
    let mut magsq = arrayfire::pow(&dir_line,&two,false);
    magsq = arrayfire::sum(&magsq, 1);
    magsq = arrayfire::sqrt(&magsq);
    
    
    
    let (std,_) = arrayfire::stdev_all_v2(&magsq,arrayfire::VarianceBias::SAMPLE);
    
    assert_eq!((std * 1000000.0).round() / 1000000.0  ,  0.0);
    
    
    
    
    
    
    let ( mut sumdist,_) = arrayfire::sum_all::<f64>(&magsq);
    
    let mut sumact: f64 = con_rad*(start_line.dims()[0] as f64);
    
    sumact = (sumact * 100000000.0).round() / 100000000.0 ;
    
    sumdist = (sumdist * 100000000.0).round() / 100000000.0 ;
    
    
    assert_eq!(sumdist, sumact);
    
    
    
    
    
    


















    let space_dims: u64 = 4;
    let con_rad: f64 = 2.8;
    
    
    
    
    
    let mut neuron_pos_cpu:Vec<f64> = vec![ 0.1, -1.7, 0.3, 0.2,          -0.1, 0.4, -0.7, -2.2,         -0.3, -4.2, -3.9, 0.9,         7.5, -0.7, -2.1, 1.2,        -0.3,-0.6,0.9,-0.11,            0.1,-0.7,0.5,-0.21,           0.2, -0.7, -5.2, 0.5 ];
    let mut neuron_pos = arrayfire::Array::new(&neuron_pos_cpu, arrayfire::Dim4::new(&[4, 7, 1, 1]));
    
    neuron_pos = arrayfire::transpose(&neuron_pos, false);
    
    let ray_num = 12;
    
    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    
    let mut start_line = arrayfire::constant::<f64>(0.0,single_dims);
    let mut dir_line = arrayfire::constant::<f64>(0.0,single_dims);
    
    RayBNN_Raytrace::Generate::Random::generate_random_uniform_rays(
        &neuron_pos,
        ray_num,
        con_rad,
    
        &mut start_line,
        &mut dir_line
        );
    
    assert_eq!(start_line.dims()[0], neuron_pos.dims()[0]*ray_num );
    assert_eq!(start_line.dims()[1], space_dims );
    
    
    assert_eq!(dir_line.dims()[0], neuron_pos.dims()[0]*ray_num );
    assert_eq!(dir_line.dims()[1], space_dims );
    
    
    
    
    
    
    
    let tile_dims = arrayfire::Dim4::new(&[ray_num,1,1,1]);
    
    let start_line_act =  arrayfire::tile(&neuron_pos, tile_dims);
    
    let mut start_line_act_cpu = vec!(f64::default();start_line_act.elements());
    start_line_act.host(&mut start_line_act_cpu);
    
    let mut start_line_pred_cpu = vec!(f64::default();start_line.elements());
    start_line.host(&mut start_line_pred_cpu);
    
    start_line_act_cpu = start_line_act_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();
    
    start_line_pred_cpu = start_line_pred_cpu.par_iter().map(|x|  (x * 1000000.0).round() / 1000000.0 ).collect::<Vec<f64>>();
    
    assert_eq!(start_line_act_cpu, start_line_pred_cpu);
    
    
    
    
    
    
    let mut magsq = arrayfire::pow(&dir_line,&two,false);
    magsq = arrayfire::sum(&magsq, 1);
    magsq = arrayfire::sqrt(&magsq);
    
    
    
    let (std,_) = arrayfire::stdev_all_v2(&magsq,arrayfire::VarianceBias::SAMPLE);
    
    assert_eq!((std * 1000000.0).round() / 1000000.0  ,  0.0);
    
    
    
    
    
    
    let ( mut sumdist,_) = arrayfire::sum_all::<f64>(&magsq);
    
    let mut sumact: f64 = con_rad*(start_line.dims()[0] as f64);
    
    sumact = (sumact * 100000000.0).round() / 100000000.0 ;
    
    sumdist = (sumdist * 100000000.0).round() / 100000000.0 ;
    
    
    assert_eq!(sumdist, sumact);
    
    
    
    




}
