#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_parse_string() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


    let inx_cpu: Vec<f64> = vec![0.0, -4.1, 1.7, -0.9, 0.3];

    let instr: String = String::from("0.0, -4.1,1.7, -0.9, 0.3");
    let outvec_cpu = RayBNN_DataLoader::Dataset::ParseString::str_to_vec_cpu::<f64>(&instr);
    assert_eq!(outvec_cpu, inx_cpu);



    let inx_cpu: Vec<i32> = vec![0, -4, 1, -9, 3];

    let instr: String = String::from("0,-4, 1,   -9,  3");
    let outvec_cpu = RayBNN_DataLoader::Dataset::ParseString::str_to_vec_cpu::<i32>(&instr);
    assert_eq!(outvec_cpu, inx_cpu);




    let inx_cpu: Vec<f32> = vec![0.0, -4.1, 1.7, -0.9, 0.3];

    let instr: String = String::from("0.0, -4.1,1.7, -0.9, 0.3");
    let outvec_cpu = RayBNN_DataLoader::Dataset::ParseString::str_to_vec_cpu::<f32>(&instr);
    assert_eq!(outvec_cpu, inx_cpu);





    let inx_cpu: Vec<i64> = vec![0, -4, 1, -9, 3];

    let instr: String = String::from("0,-4, 1,   -9,  3");
    let outvec_cpu = RayBNN_DataLoader::Dataset::ParseString::str_to_vec_cpu::<i64>(&instr);
    assert_eq!(outvec_cpu, inx_cpu);



    let inx_cpu: Vec<u64> = vec![0, 4, 1, 9, 3];

    let instr: String = String::from("0, 4, 1, 9, 3");
    let outvec_cpu = RayBNN_DataLoader::Dataset::ParseString::str_to_vec_cpu::<u64>(&instr);
    assert_eq!(outvec_cpu, inx_cpu);


    let inx_cpu: Vec<u32> = vec![0, 4, 1, 9, 3];

    let instr: String = String::from("0, 4, 1, 9, 3");
    let outvec_cpu = RayBNN_DataLoader::Dataset::ParseString::str_to_vec_cpu::<u32>(&instr);
    assert_eq!(outvec_cpu, inx_cpu);



    let instr: String = String::from("aaa, bbb, ccc");
    let outvec_cpu = RayBNN_DataLoader::Dataset::ParseString::str_to_vec_cpu::<u64>(&instr);
    assert_eq!(outvec_cpu.len(), 0);
















    let inx_cpu: Vec<i32> = vec![0, -4, 1, -9,   3];

    let outstr: String = String::from("0,-4,1,-9,3");
    let outstr_cpu = RayBNN_DataLoader::Dataset::ParseString::vec_cpu_to_str::<i32>(&inx_cpu);
    assert_eq!(outstr_cpu, outstr);





    let inx_cpu: Vec<i64> = vec![0, -4, 1, -9,   3];

    let outstr: String = String::from("0,-4,1,-9,3");
    let outstr_cpu = RayBNN_DataLoader::Dataset::ParseString::vec_cpu_to_str::<i64>(&inx_cpu);
    assert_eq!(outstr_cpu, outstr);



    let inx_cpu: Vec<f64> = vec![0.0, -4.1,1.7, -0.9, 0.3];

    let outstr: String = String::from("0.0,-4.1,1.7,-0.9,0.3");
    let outstr_cpu = RayBNN_DataLoader::Dataset::ParseString::vec_cpu_to_str::<f64>(&inx_cpu);
    assert_eq!(outstr_cpu, outstr);



    let inx_cpu: Vec<f32> = vec![0.0, -4.1,1.7, -0.9, 0.3];

    let outstr: String = String::from("0.0,-4.1,1.7,-0.9,0.3");
    let outstr_cpu = RayBNN_DataLoader::Dataset::ParseString::vec_cpu_to_str::<f32>(&inx_cpu);
    assert_eq!(outstr_cpu, outstr);








    let inx_cpu: Vec<u32> = vec![0, 4, 1, 9,   3];

    let outstr: String = String::from("0,4,1,9,3");
    let outstr_cpu = RayBNN_DataLoader::Dataset::ParseString::vec_cpu_to_str::<u32>(&inx_cpu);
    assert_eq!(outstr_cpu, outstr);



    let inx_cpu: Vec<u64> = vec![0, 4, 1, 9,   3];

    let outstr: String = String::from("0,4,1,9,3");
    let outstr_cpu = RayBNN_DataLoader::Dataset::ParseString::vec_cpu_to_str::<u64>(&inx_cpu);
    assert_eq!(outstr_cpu, outstr);









}
