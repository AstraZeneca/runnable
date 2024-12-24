import datetime
import json
import logging
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Union, cast

from runnable import defaults, utils
from runnable.extensions.run_log_store.generic_chunked import ChunkedRunLogStore

logger = logging.getLogger(defaults.LOGGER_NAME)


class DBRunLogStore(ChunkedRunLogStore):
    """
    File system run log store but chunks the run log into thread safe chunks.
    This enables executions to be parallel.
    """

    service_name: str = "chunked-fs"
    connection_string: str
    db_name: str

    _DB_LOG: Any = None
    _engine: Any = None
    _session: Any = None
    _connection_string: str = ""
    _base: Any = None

    def model_post_init(self, _: Any) -> None:
        run_context = self._context

        secrets = cast(Dict[str, str], run_context.secrets_handler.get())
        connection_string = Template(self.connection_string).safe_substitute(**secrets)

        try:
            import sqlalchemy
            from sqlalchemy import Column, DateTime, Integer, Sequence, Text
            from sqlalchemy.orm import declarative_base, sessionmaker

            Base = declarative_base()

            class DBLog(Base):
                """
                Base table for storing run logs in database.

                In this model, we fragment the run log into logical units that are concurrent safe.
                """

                __tablename__ = self.db_name
                pk = Column(Integer, Sequence("id_seq"), primary_key=True)
                run_id = Column(Text, index=True)
                attribute_key = Column(
                    Text
                )  # run_log, step_internal_name, parameter_key etc
                attribute_type = Column(Text)  # RunLog, Step, Branch, Parameter
                attribute_value = Column(Text)  # The JSON string
                created_at = Column(DateTime, default=datetime.datetime.utcnow)

            self._engine = sqlalchemy.create_engine(
                connection_string, pool_pre_ping=True
            )
            self._session = sessionmaker(bind=self._engine)
            self._DB_LOG = DBLog
            self._connection_string = connection_string
            self._base = Base

        except ImportError as _e:
            logger.exception("Unable to import SQLalchemy, is it installed?")
            msg = "SQLAlchemy is required for this extension. Please install it"
            raise Exception(msg) from _e

    def create_tables(self):
        import sqlalchemy

        engine = sqlalchemy.create_engine(self._connection_string)
        self._base.metadata.create_all(engine)

    def get_matches(
        self, run_id: str, name: str, multiple_allowed: bool = False
    ) -> Optional[Union[List[Path], Path]]:
        """
        Get contents of files matching the pattern name*

        Args:
            run_id (str): The run id
            name (str): The suffix of the file name to check in the run log store.
        """
        log_folder = self.log_folder_with_run_id(run_id=run_id)

        sub_name = Template(name).safe_substitute({"creation_time": ""})

        matches = list(log_folder.glob(f"{sub_name}*"))
        if matches:
            if not multiple_allowed:
                if len(matches) > 1:
                    msg = f"Multiple matches found for {name} while multiple is not allowed"
                    raise Exception(msg)
                return matches[0]
            return matches

        return None

    def log_folder_with_run_id(self, run_id: str) -> Path:
        """
        Utility function to get the log folder for a run id.

        Args:
            run_id (str): The run id

        Returns:
            Path: The path to the log folder with the run id
        """
        return Path(self.log_folder) / run_id

    def safe_suffix_json(self, name: Union[Path, str]) -> str:
        """
        Safely attach a suffix to a json file.

        Args:
            name (Path): The name of the file with or without suffix of json

        Returns:
            str : The name of the file with .json
        """
        if str(name).endswith("json"):
            return str(name)

        return str(name) + ".json"

    def _store(self, run_id: str, contents: dict, name: Union[Path, str], insert=False):
        """
        Store the contents against the name in the folder.

        Args:
            run_id (str): The run id
            contents (dict): The dict to store
            name (str): The name to store as
        """
        if insert:
            name = self.log_folder_with_run_id(run_id=run_id) / name

        utils.safe_make_dir(self.log_folder_with_run_id(run_id=run_id))

        with open(self.safe_suffix_json(name), "w") as fw:
            json.dump(contents, fw, ensure_ascii=True, indent=4)

    def _retrieve(self, name: Path) -> dict:
        """
        Does the job of retrieving from the folder.

        Args:
            name (str): the name of the file to retrieve

        Returns:
            dict: The contents
        """
        contents: dict = {}

        with open(self.safe_suffix_json(name), "r") as fr:
            contents = json.load(fr)

        return contents
