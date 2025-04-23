// mymodule.c
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// C 함수: 두 정수를 더함
int add(int a, int b) {
    return a + b;
}

// Python에서 호출할 수 있는 래퍼 함수
static PyObject* py_add(PyObject* self, PyObject* args) {
    int a, b;
    // Python에서 전달된 인수를 파싱
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL; // 파싱 실패 시 NULL 반환
    }
    int result = add(a, b);
    // 결과를 Python 객체로 반환
    return PyLong_FromLong(result);
}

// 메서드 정의 구조체
static PyMethodDef MyModuleMethods[] = {
    {"add", py_add, METH_VARARGS, "Add two integers."},
    {NULL, NULL, 0, NULL} // 종료를 알림
};

// 모듈 정의 구조체
static struct PyModuleDef mymodulemodule = {
    PyModuleDef_HEAD_INIT,
    "mymodule", // 모듈 이름
    "A simple C extension module.", // 모듈 설명
    -1,
    MyModuleMethods
};

// 모듈 초기화 함수
PyMODINIT_FUNC PyInit_mymodule(void) {
    return PyModule_Create(&mymodulemodule);
}
