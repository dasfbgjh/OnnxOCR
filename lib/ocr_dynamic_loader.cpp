#include "ocr_dynamic_loader.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstring>

namespace ocr {

// ---- 平台辅助函数 ----

#ifdef _WIN32

static std::string extract_dir( const std::string &path ) {
    size_t pos = path.find_last_of( "/\\" );
    if ( pos == std::string::npos )
        return ".";
    return path.substr( 0, pos );
}

static void *platform_load( const std::string &path ) {
    std::string p = path.empty() ? std::string( "OnnxOCR.dll" ) : path;
    std::string dir = extract_dir( p );

    std::wstring wDir( dir.begin(), dir.end() );

    HMODULE h = nullptr;

#if _WIN32_WINNT >= 0x0602
    typedef DLL_DIRECTORY_COOKIE( WINAPI * AddDllDirectory_t )( PCWSTR );
    typedef BOOL( WINAPI * RemoveDllDirectory_t )( DLL_DIRECTORY_COOKIE );

    HMODULE k32 = GetModuleHandleA( "kernel32.dll" );
    auto pAdd = reinterpret_cast<AddDllDirectory_t>(
        GetProcAddress( k32, "AddDllDirectory" ) );
    auto pRemove = reinterpret_cast<RemoveDllDirectory_t>(
        GetProcAddress( k32, "RemoveDllDirectory" ) );

    if ( pAdd ) {
        DLL_DIRECTORY_COOKIE cookie = pAdd( wDir.c_str() );
        if ( cookie == 0 )
            return nullptr;
        h = LoadLibraryExA( p.c_str(), nullptr,
                            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS );
        if ( !h && pRemove )
            pRemove( cookie );
    } else {
        std::string oldDir;
        char buf[MAX_PATH] = {};
        if ( GetDllDirectoryA( MAX_PATH, buf ) )
            oldDir = buf;
        SetDllDirectoryA( dir.c_str() );
        h = LoadLibraryA( p.c_str() );
        SetDllDirectoryA( oldDir.empty() ? nullptr : oldDir.c_str() );
    }
#else
    std::string oldDir;
    char buf[MAX_PATH] = {};
    if ( GetDllDirectoryA( MAX_PATH, buf ) )
        oldDir = buf;
    SetDllDirectoryA( dir.c_str() );
    h = LoadLibraryA( p.c_str() );
    SetDllDirectoryA( oldDir.empty() ? nullptr : oldDir.c_str() );
#endif

    return reinterpret_cast<void *>( h );
}

static void platform_unload( void *handle ) {
    if ( handle )
        FreeLibrary( reinterpret_cast<HMODULE>( handle ) );
}

static void *platform_resolve( void *handle, const char *name ) {
    if ( !handle )
        return nullptr;
    return reinterpret_cast<void *>( GetProcAddress( reinterpret_cast<HMODULE>( handle ), name ) );
}

static std::string platform_error() {
    DWORD err = GetLastError();
    if ( err == 0 )
        return "no error";
    LPSTR buf = nullptr;
    FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, err, MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT ),
        reinterpret_cast<LPSTR>( &buf ), 0, nullptr );
    std::string msg( buf ? buf : "unknown error" );
    if ( buf )
        LocalFree( buf );
    return msg;
}
#else
static void *platform_load( const std::string &path ) {
    std::string p;
    if ( path.empty() ) {
#if defined( __APPLE__ )
        p = "libOnnxOCR.dylib";
#else
        p = "libOnnxOCR.so";
#endif
    } else {
        p = path;
    }
    return dlopen( p.c_str(), RTLD_NOW | RTLD_GLOBAL );
}

static void platform_unload( void *handle ) {
    if ( handle )
        dlclose( handle );
}

static void *platform_resolve( void *handle, const char *name ) {
    if ( !handle )
        return nullptr;
    return dlsym( handle, name );
}

static std::string platform_error() {
    const char *err = dlerror();
    return err ? err : "no error";
}
#endif

// ---- OcrDynamicLoader ----

OcrDynamicLoader::OcrDynamicLoader( const std::string &lib_path ) {
    lib_handle_ = platform_load( lib_path );
    if ( !lib_handle_ ) {
        throw DynamicLoaderError(
            "failed to load OnnxOCR library '" +
            ( lib_path.empty() ? std::string( "default" ) : lib_path ) +
            "': " + platform_error() );
    }

    try {
        resolve_symbols();
    } catch ( ... ) {
        platform_unload( lib_handle_ );
        lib_handle_ = nullptr;
        throw;
    }
}

OcrDynamicLoader::~OcrDynamicLoader() {
    if ( lib_handle_ ) {
        platform_unload( lib_handle_ );
        lib_handle_ = nullptr;
    }
}

void OcrDynamicLoader::resolve_symbols() {
    struct SymEntry {
        const char *name;
        void **target;
    };
    SymEntry entries[] = {
        { "last_error", reinterpret_cast<void **>( &last_error_fn_ ) },

        // OCR API
        { "ocr_create", reinterpret_cast<void **>( &ocr_create_fn_ ) },
        { "ocr_destroy", reinterpret_cast<void **>( &ocr_destroy_fn_ ) },
        { "ocr_run", reinterpret_cast<void **>( &ocr_run_fn_ ) },
        { "ocr_run_file", reinterpret_cast<void **>( &ocr_run_file_fn_ ) },
        { "ocr_free_results", reinterpret_cast<void **>( &ocr_free_results_fn_ ) },

        // 车牌识别API
        { "plate_create", reinterpret_cast<void **>( &plate_create_fn_ ) },
        { "plate_destroy", reinterpret_cast<void **>( &plate_destroy_fn_ ) },
        { "plate_run", reinterpret_cast<void **>( &plate_run_fn_ ) },
        { "plate_run_file", reinterpret_cast<void **>( &plate_run_file_fn_ ) },
        { "plate_free_results", reinterpret_cast<void **>( &plate_free_results_fn_ ) },

        // 表格识别API
        { "table_create", reinterpret_cast<void **>( &table_create_fn_ ) },
        { "table_destroy", reinterpret_cast<void **>( &table_destroy_fn_ ) },
        { "table_run", reinterpret_cast<void **>( &table_run_fn_ ) },
        { "table_run_file", reinterpret_cast<void **>( &table_run_file_fn_ ) },
        { "table_free_result", reinterpret_cast<void **>( &table_free_result_fn_ ) },

        // 版面分析统一API
        { "doclayout_yolo_create", reinterpret_cast<void **>( &doclayout_yolo_create_fn_ ) },
        { "doclayout_yolo_destroy", reinterpret_cast<void **>( &doclayout_yolo_destroy_fn_ ) },
        { "doclayout_yolo_run", reinterpret_cast<void **>( &doclayout_yolo_run_fn_ ) },
        { "doclayout_yolo_run_file", reinterpret_cast<void **>( &doclayout_yolo_run_file_fn_ ) },
        { "doc_layout_free_results", reinterpret_cast<void **>( &doc_layout_free_results_fn_ ) },

        // 图像显示API
        { "image_show_rects", reinterpret_cast<void **>( &image_show_rects_fn_ ) },
        { "image_show_rects_file", reinterpret_cast<void **>( &image_show_rects_file_fn_ ) },

    };

    for ( const auto &e : entries ) {
        *e.target = platform_resolve( lib_handle_, e.name );
        if ( !*e.target ) {
            throw DynamicLoaderError(
                std::string( "failed to resolve symbol '" ) + e.name +
                "': " + platform_error() );
        }
    }
}

} // namespace ocr