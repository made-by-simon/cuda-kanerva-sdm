#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kanerva_sdm.h"

namespace py = pybind11;

PYBIND11_MODULE(sdm_cpp, m) {
    m.doc() = "C++ implementation of Kanerva's Sparse Distributed Memory";

    py::class_<KanervaSDM>(m, "KanervaSDM")
        .def(py::init<int, int, int, int, unsigned int>(),
             py::arg("address_dimension"),
             py::arg("memory_dimension"),
             py::arg("num_locations"),
             py::arg("activation_threshold"),
             py::arg("random_seed") = 42,
             "Initialize Kanerva SDM")
        
        .def("write", &KanervaSDM::write,
             py::arg("address"),
             py::arg("memory"),
             "Write a memory to an address")
        
        .def("read", &KanervaSDM::read,
             py::arg("address"),
             "Read a memory from an address")
        
        .def("erase_memory", &KanervaSDM::erase_memory,
             "Erase memory matrix but preserve hard locations")
        
        .def_property_readonly("address_dimension", &KanervaSDM::get_address_dimension,
                               "Address dimension (N)")

        .def_property_readonly("memory_dimension", &KanervaSDM::get_memory_dimension,
                               "Memory dimension (U)")

        .def_property_readonly("num_locations", &KanervaSDM::get_num_locations,
                               "Number of hard locations (M)")

        .def_property_readonly("activation_threshold", &KanervaSDM::get_activation_threshold,
                               "Activation threshold (H)")

        .def_property_readonly("memory_count", &KanervaSDM::get_memory_count,
                               "Number of stored memories (T)");
}