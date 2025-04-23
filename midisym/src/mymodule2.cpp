#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject* say_hello(PyObject* self, PyObject* args) {
    return Py_BuildValue("s", "Hello from C++!");
}

static PyMethodDef SampleMethods[] = {
    {"say_hello", say_hello, METH_VARARGS, "Prints a hello message."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef samplemodule = {
    PyModuleDef_HEAD_INIT,
    "mymodule2",
    NULL,
    -1,
    SampleMethods
};

PyMODINIT_FUNC PyInit_mymodule2(void) {
    import_array();  // Required for numpy
    return PyModule_Create(&samplemodule);
}
