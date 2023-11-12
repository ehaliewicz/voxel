//author https://github.com/autergame
#pragma once

#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <windows.h>
#include <synchapi.h>
#include <winternl.h>

#pragma comment(lib, "ntdll")


typedef struct thread_pool_
{
    TP_CALLBACK_ENVIRON callback_environ;
    PTP_CLEANUP_GROUP cleanup_group;
    PTP_POOL pool;
} thread_pool;

#define thread_pool_function(function_name, arg_var_name) \
    void CALLBACK function_name(PTP_CALLBACK_INSTANCE instance, PVOID arg_var_name, PTP_WORK work)



thread_pool* thread_pool_create(int cpu_threads)
{
    assert(cpu_threads > 0);
    thread_pool* tp = (thread_pool*)calloc(1, sizeof(thread_pool));

    if (tp) {


        InitializeThreadpoolEnvironment(&tp->callback_environ);

        tp->pool = CreateThreadpool(NULL);

        SetThreadpoolThreadMinimum(tp->pool, cpu_threads);
        SetThreadpoolThreadMaximum(tp->pool, cpu_threads);

        tp->cleanup_group = CreateThreadpoolCleanupGroup();

        SetThreadpoolCallbackPool(&tp->callback_environ, tp->pool);
        SetThreadpoolCallbackCleanupGroup(&tp->callback_environ, tp->cleanup_group, NULL);
    }

    return tp;
}

void thread_pool_add_work(thread_pool* tp, PTP_WORK_CALLBACK function, void* arg_var)
{
    if (tp)
    {
        PTP_WORK work = CreateThreadpoolWork(function, arg_var, &tp->callback_environ);
        SubmitThreadpoolWork(work);
    }
}

void thread_pool_destroy(thread_pool* tp)
{
    if (tp)
    {
        CloseThreadpoolCleanupGroupMembers(tp->cleanup_group, FALSE, NULL);
        CloseThreadpoolCleanupGroup(tp->cleanup_group);

        DestroyThreadpoolEnvironment(&tp->callback_environ);

        CloseThreadpool(tp->pool);

        free(tp);
    }
}