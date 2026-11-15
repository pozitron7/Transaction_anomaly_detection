"""
Apache Spark Session & Parquet I/O Utilities.

Provides centralized SparkSession lifecycle management with adaptive query execution,
Arrow optimization, automatic JDK 17/21 detection on Windows (when default JDK is 22+),
and cross-platform Parquet writing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession


def _ensure_compatible_java_home() -> None:
    """
    Apache Spark / Hadoop 3.x requires Java 8, 11, 17, or 21 (Java 24+ removed
    Subject.getSubject). If JAVA_HOME is unset or points to Java 22+ on Windows,
    automatically select an installed JDK 17 or 21 if available.
    """
    if os.name != "nt":
        return

    def _java_major(java_exe: str) -> Optional[int]:
        try:
            out = subprocess.check_output(
                [java_exe, "-version"], stderr=subprocess.STDOUT, text=True
            )
            for line in out.splitlines():
                if "version" in line:
                    ver = line.split('"')[1].split(".")[0]
                    return int(ver)
        except Exception:
            return None
        return None

    current_java = (
        str(Path(os.environ["JAVA_HOME"]) / "bin" / "java.exe")
        if os.environ.get("JAVA_HOME")
        else "java"
    )
    major = _java_major(current_java)
    if major is not None and 8 <= major <= 21:
        return

    search_roots = [
        Path("C:/Program Files/Eclipse Adoptium"),
        Path("C:/Program Files/Java"),
        Path("C:/Program Files/Microsoft"),
        Path("C:/Program Files/Zulu"),
        Path("C:/Program Files/Amazon Corretto"),
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in sorted(root.iterdir(), reverse=True):
            java_bin = candidate / "bin" / "java.exe"
            if java_bin.exists():
                cand_major = _java_major(str(java_bin))
                if cand_major is not None and 8 <= cand_major <= 21:
                    os.environ["JAVA_HOME"] = str(candidate)
                    os.environ["PATH"] = f"{candidate / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
                    return


def get_spark_session(config: Optional[Dict[str, Any]] = None) -> SparkSession:
    """Create or retrieve a configured Apache SparkSession."""
    _ensure_compatible_java_home()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    spark_cfg = (config or {}).get("spark", {})
    app_name = spark_cfg.get("app_name", "TransactionAnomalyDetection")
    master = spark_cfg.get("master", "local[*]")
    driver_memory = spark_cfg.get("driver_memory", "4g")
    shuffle_partitions = str(spark_cfg.get("shuffle_partitions", 32))
    adaptive_enabled = str(spark_cfg.get("adaptive_enabled", True)).lower()

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.sql.adaptive.enabled", adaptive_enabled)
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def write_spark_parquet(df: DataFrame, output_path: Path) -> None:
    """
    Persist a Spark DataFrame to Parquet.

    Uses Spark's native distributed Parquet writer when supported (Linux/macOS or
    Windows with HADOOP_HOME/winutils.exe configured). Falls back cleanly to PyArrow
    Parquet writing on Windows environments lacking winutils.exe.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    can_use_hadoop_fs = os.name != "nt" or bool(os.environ.get("HADOOP_HOME"))
    if can_use_hadoop_fs:
        df.write.mode("overwrite").parquet(str(output_path))
        return

    # Windows without winutils.exe: remove any previous file/directory cleanly and
    # write via PyArrow (streaming via toArrow() when available).
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    if hasattr(df, "toArrow"):
        table = df.toArrow()
    else:
        table = pa.Table.from_pandas(df.toPandas(), preserve_index=False)

    pq.write_table(table, str(output_path), compression="snappy")
