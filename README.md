# RayBNN_DataLoader
Data Loader for RayBNN

Read CSV, numpy, and binary files to Rust vectors of f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64

Read CSV, numpy, and binary files to Arrayfire GPU arrays of f16, f32, f64, u8, u16, u32, u64, i8, i16, i32, i64


# Install Arrayfire

Install the Arrayfire 3.9.0 binaries at [https://arrayfire.com/binaries/](https://arrayfire.com/binaries/)

or build from source
[https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire](https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire)




# Add to Cargo.toml
```
arrayfire = { version = "3.8.1", package = "arrayfire_fork" }
rayon = "1.7.0"
num = "0.4.1"
num-traits = "0.2.16"
half = { version = "2.3.1" , features = ["num-traits"] }
npyz = "0.8.1"
RayBNN_DataLoader = "0.1.4"
```

# List of Examples



# Read a CSV file to a floating point 64 bit CPU Vector
The vector is completely flat
```
let (mut cpu_vector,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<f64>(
    "./test_data/read_test.dat"
);
```

# Read a CSV file to a integer 64 bit CPU Vector
The vector is completely flat
```
let (mut cpu_vector,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_vec_cpu::<i64>(
    "./test_data/read_test2.dat"
);
```

# Read a CSV file to a floating point 64 bit arrayfire
The array is 2D existing in GPU or OpenCL
```
let read_test = RayBNN_DataLoader::Dataset::CSV::file_to_arrayfire::<f64>(
    "./test_data/read_test.dat"
);
```


# Read a CSV file to a floating point 64 bit HashMap
```
let hashdata = RayBNN_DataLoader::Dataset::CSV::file_to_hash_cpu::<f64>(
    "./test_data/dataloader.csv",
    7,
    5
);
```

# Read a CSV file to a floating point 64 bit HashMap with Arrayfire
```
let arr_dims = arrayfire::Dim4::new(&[5, 7, 1, 1]);
let (hashdata,metadata) = RayBNN_DataLoader::Dataset::CSV::file_to_hash_arrayfire::<f64>(
    "./test_data/dataloader.csv",
    arr_dims,
);
```

# Write a float 32 bit CPU vector to CSV file
```
let mut metadata: HashMap<String,u64> = HashMap::new();

metadata.insert("dim0", 11);
metadata.insert("dim1", 3);

RayBNN_DataLoader::Dataset::CSV::write_vec_cpu_to_csv::<f32>(
    "./randvec2.csv",
    &randvec,
    &metadata
);
```



# Write a float 64 bit arrayfire to CSV file
```
RayBNN_DataLoader::Dataset::CSV::write_arrayfire_to_csv::<f64>(
    "./randvec.csv",
    &arr
);
```



# Read YAML Model Information
```
let mut modeldata_string:  HashMap<String, String> = HashMap::new();
let mut modeldata_float:  HashMap<String, f64> = HashMap::new();
let mut modeldata_int:  HashMap<String, u64> = HashMap::new();

RayBNN_DataLoader::Model::YAML::read(
    "./test_data/test.yaml",

    &mut modeldata_string,
    &mut modeldata_float,
    &mut modeldata_int,
);
```



# Write YAML Model Information
```
RayBNN_DataLoader::Model::YAML::write(
    "./sample.yaml",

    &modeldata_string,
    &modeldata_float,
    &modeldata_int,
);
```