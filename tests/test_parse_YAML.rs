#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;

use std::collections::HashMap;


#[test]
fn test_parse_YAML() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


    let mut modeldata_string:  HashMap<String, String> = HashMap::new();
    let mut modeldata_float:  HashMap<String, f64> = HashMap::new();
    let mut modeldata_int:  HashMap<String, u64> = HashMap::new();

    RayBNN_DataLoader::Model::YAML::read(
        "./test_data/test.yaml",
    
        &mut modeldata_string,
        &mut modeldata_float,
        &mut modeldata_int,
    );

    assert!(modeldata_string.contains_key("model_filename"));
    assert_eq!(modeldata_string["model_filename"].clone(), "/opt/test/");


    assert!(modeldata_string.contains_key("data_filename"));
    assert_eq!(modeldata_string["data_filename"].clone(), "/tmp/test/");


    assert!(modeldata_float.contains_key("version"));
    assert_eq!(modeldata_float["version"].clone(), 1.5);

    assert!(modeldata_float.contains_key("add_ratio"));
    assert_eq!(modeldata_float["add_ratio"].clone(), 4.7);


    assert!(modeldata_int.contains_key("active_size"));
    assert_eq!(modeldata_int["active_size"].clone(), 1552);

    assert!(modeldata_int.contains_key("input_size"));
    assert_eq!(modeldata_int["input_size"].clone(), 15);














    let mut modeldata_string:  HashMap<String, String> = HashMap::new();
    let mut modeldata_float:  HashMap<String, f64> = HashMap::new();
    let mut modeldata_int:  HashMap<String, u64> = HashMap::new();

    modeldata_int.insert("num_neurons".to_string(), 65612);
    modeldata_int.insert("num_glia".to_string(), 8034);
    
    modeldata_float.insert("growth_rate".to_string(), 3.2);
    modeldata_float.insert("delete_rate".to_string(), 0.1);

    modeldata_string.insert("model_name".to_string(), "laser".to_string());
    modeldata_string.insert("type".to_string(), "RayBNN".to_string());
    
    RayBNN_DataLoader::Model::YAML::write(
        "./sample.yaml",
    
        &modeldata_string,
        &modeldata_float,
        &modeldata_int,
    );
    drop(modeldata_string);
    drop(modeldata_float);
    drop(modeldata_int);


    let mut modeldata_string:  HashMap<String, String> = HashMap::new();
    let mut modeldata_float:  HashMap<String, f64> = HashMap::new();
    let mut modeldata_int:  HashMap<String, u64> = HashMap::new();

    RayBNN_DataLoader::Model::YAML::read(
        "./sample.yaml",
    
        &mut modeldata_string,
        &mut modeldata_float,
        &mut modeldata_int,
    );

    assert!(modeldata_string.contains_key("model_name"));
    assert_eq!(modeldata_string["model_name"].clone(), "laser");

    assert!(modeldata_string.contains_key("type"));
    assert_eq!(modeldata_string["type"].clone(), "RayBNN");

    assert!(modeldata_int.contains_key("num_neurons"));
    assert_eq!(modeldata_int["num_neurons"].clone(), 65612);

    assert!(modeldata_int.contains_key("num_glia"));
    assert_eq!(modeldata_int["num_glia"].clone(), 8034);

    assert!(modeldata_float.contains_key("growth_rate"));
    assert_eq!(modeldata_float["growth_rate"].clone(), 3.2);

    assert!(modeldata_float.contains_key("delete_rate"));
    assert_eq!(modeldata_float["delete_rate"].clone(), 0.1);



    std::fs::remove_file("./sample.yaml");
}
