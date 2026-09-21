"""unitree SDK の interface 指定 config から <Tracing> を落とす。

cyclonedds 0.10.2 は `<Tracing><Verbosity>config</Verbosity></Tracing>` を含む
config で Domain を作ると glibc の _FORTIFY_SOURCE に引っかかって
`*** buffer overflow detected ***` で core dump する (image の中で実測)。
unitree の ChannelConfigHasInterface は必ずこの block を持つので、
`ChannelFactoryInitialize(0, <interface>)` = 自前経路の DDS 初期化が全滅する。

Tracing は /tmp/cdds.LOG への診断出力なので、落としても通信には影響しない。
interface 指定のない ChannelConfigAutoDetermine は元から Tracing を持たず、
そちらは落ちない (これが「interface を渡したときだけ落ちる」理由)。
"""

import sys
from pathlib import Path

TARGET = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/opt/unitree_sdk2_python/unitree_sdk2py/core/channel_config.py"
)
BLOCK = """            <Tracing>
                <Verbosity>config</Verbosity>
            <OutputFile>/tmp/cdds.LOG</OutputFile>
        </Tracing>
"""

s = TARGET.read_text(encoding="utf-8")
if BLOCK not in s:
    if "<Tracing>" in s:
        print("[patch] FAILED: Tracing block の形が変わっている", file=sys.stderr)
        sys.exit(1)
    print("[patch] already-applied")
    sys.exit(0)
TARGET.write_text(s.replace(BLOCK, ""), encoding="utf-8")
print("[patch] applied: removed <Tracing> from ChannelConfigHasInterface")
