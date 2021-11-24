/* 
推荐总是在#include <Python.h>前定义这个宏
*/
#define PY_SSIZE_T_CLEAN	

/*
要在包含其他标准库头文件之前包含Python.h
里面已经包含了<stdio.h><string.h><errno.h><stdlib.h>
*/
#include <Python.h>	
#include "func.h"

/*
模块内的函数实现, 也就是要使用的C语言函数
可以有static修饰
函数形式:
    PyObject* MyFunction(PyObject* self);
    PyObject* MyFunction(PyObject* self, PyObject* args);
    PyObject* MyFunction(PyObject* self, PyObject* args, PyObject* kws);
    推荐函数名前加上模块名前缀用于C代码的区分
    self是必须有的参数, 指向模块本身, 剩余的参数与python形式的收集参数和关键字参数相同
    需要使用相应的API进行参数解析
    PyArg_ParseTuple
    PyArg_ParseTupleAndKeywords
*/
static PyObject* pyext_add(PyObject* self, PyObject* args)
{
    double a, b, result;
    if (!PyArg_ParseTuple(args, "dd:myadd", &a, &b))
        return NULL;

    result = myadd(a, b);
    return PyFloat_FromDouble(result);
}

static PyObject* pyext_sub(PyObject* self, PyObject* args)
{
    double a, b, result;
    if (!PyArg_ParseTuple(args, "dd:mysub", &a, &b))
        return NULL;

    result = mysub(a, b);;
    return PyFloat_FromDouble(result);
}

static PyObject* pyext_mul(PyObject* self, PyObject* args)
{
    double a, b, result;
    if (!PyArg_ParseTuple(args, "dd:mymul", &a, &b))
        return NULL;

    result = mymul(a, b);
    return PyFloat_FromDouble(result);
}

static PyObject* pyext_div(PyObject* self, PyObject* args)
{
    double a, b, result;
    if (!PyArg_ParseTuple(args, "dd:mydiv", &a, &b))
        return NULL;

    result = mydiv(a, b);
    return PyFloat_FromDouble(result);
}

/*
模块方法定义表:
    PyMethodDef ModuleMethods[];
    {python中的函数名, C代码中的入口地址, 调用方式, __doc__}
*/
static PyMethodDef pyext_Methods[] = {
    { "add", (PyCFunction)pyext_add, METH_VARARGS, "add two num" },
    { "sub", (PyCFunction)pyext_sub, METH_VARARGS, "sub two num" },
    { "mul", (PyCFunction)pyext_mul, METH_VARARGS, "mul two num" },
    { "div", (PyCFunction)pyext_div, METH_VARARGS, "div two num" },
    { NULL, NULL, 0, NULL }
};

/*
模块定义结构:
    static PyModuleDef Module;
    该结构体要被传递给解释器的模块初始化函数
    {PyModuleDef_HEAD_INIT, <ModuleName>, __doc__, 0, 模块方法表}
*/
static PyModuleDef pyext_Module = {
    PyModuleDef_HEAD_INIT,
    "pyext",
    "provide functions implemented via C",
    0,
    pyext_Methods
};

/*
模块初始化函数:
    命名方式: PyInit_<ModuleName>, 必须是如此形式且非static函数, <ModuleName>需要与文件名相同
    PyMODINIT_FUNC PyInit_<ModuleName>();
*/
PyMODINIT_FUNC PyInit_pyext() {
    return PyModule_Create(&pyext_Module);
}

