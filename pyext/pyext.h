#pragma once
#ifdef PYEXT_EXPORTS
#define PYEXT_API __declspec(dllexport)
#else
#define PYEXT_API __declspec(dllimport)
#endif // PYEXT_EXPORT
