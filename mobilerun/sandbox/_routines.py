"""Optional routines dependency exports for sandbox scripts."""

try:
    from mobilerun_routines import (
        ALLOWED_FORMATS as ALLOWED_FORMATS,
    )
    from mobilerun_routines import (
        DATASET_MAX_BYTES as DATASET_MAX_BYTES,
    )
    from mobilerun_routines import (
        ERROR_MESSAGE_MAX_CHARS as ERROR_MESSAGE_MAX_CHARS,
    )
    from mobilerun_routines import (
        RESERVED_ERROR_CODES as RESERVED_ERROR_CODES,
    )
    from mobilerun_routines import (
        RESULT_MAX_BYTES as RESULT_MAX_BYTES,
    )
    from mobilerun_routines import (
        STATE_MAX_BYTES as STATE_MAX_BYTES,
    )
    from mobilerun_routines import (
        Context as Context,
    )
    from mobilerun_routines import (
        Dataset as Dataset,
    )
    from mobilerun_routines import (
        DatasetItemInvalidError as DatasetItemInvalidError,
    )
    from mobilerun_routines import (
        DatasetTooLargeError as DatasetTooLargeError,
    )
    from mobilerun_routines import (
        EntryPointError as EntryPointError,
    )
    from mobilerun_routines import (
        Log as Log,
    )
    from mobilerun_routines import (
        NotAvailableError as NotAvailableError,
    )
    from mobilerun_routines import (
        RunInfo as RunInfo,
    )
    from mobilerun_routines import (
        ScriptError as ScriptError,
    )
    from mobilerun_routines import (
        key_field as key_field,
    )
    from mobilerun_routines import (
        script as script,
    )
except ModuleNotFoundError as exc:
    if exc.name != "mobilerun_routines":
        raise
    raise ImportError(
        "Sandbox support requires the optional dependency; "
        "install it with pip install 'mobilerun[sandbox]'"
    ) from exc
