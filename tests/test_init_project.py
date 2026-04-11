"""
Testes unitários para init_project.ProjectInitializer.

Cobre:
    - Criação de diretórios quando não existem
    - Idempotência (executar múltiplas vezes sem erro)
    - Lista de diretórios configurada corretamente
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from init_project import ProjectInitializer


class TestProjectInitializer:
    """Testes da classe ProjectInitializer."""

    def test_directories_configurados(self):
        """Deve ter lista de diretórios não vazia."""
        initializer = ProjectInitializer()
        assert len(initializer.directories) > 0

    def test_directories_sao_paths(self):
        """Todos os diretórios devem ser instâncias de Path."""
        initializer = ProjectInitializer()
        for d in initializer.directories:
            assert isinstance(d, Path), f"{d} não é Path"

    def test_create_directories(self, tmp_path, monkeypatch):
        """Deve criar todos os diretórios configurados."""
        dirs = [
            tmp_path / "dir1",
            tmp_path / "dir2" / "subdir",
        ]
        initializer = ProjectInitializer()
        monkeypatch.setattr(initializer, "directories", dirs)
        initializer.create_directories()

        for d in dirs:
            assert d.exists()

    def test_create_directories_idempotente(self, tmp_path, monkeypatch):
        """Executar create_directories duas vezes não deve gerar erro."""
        dirs = [tmp_path / "exist"]
        initializer = ProjectInitializer()
        monkeypatch.setattr(initializer, "directories", dirs)
        initializer.create_directories()
        initializer.create_directories()  # Segunda vez — sem erro

        assert dirs[0].exists()

    def test_run_executa_sem_erro(self, tmp_path, monkeypatch):
        """O método run deve executar o pipeline completo sem exceções."""
        dirs = [tmp_path / "run_test"]
        initializer = ProjectInitializer()
        monkeypatch.setattr(initializer, "directories", dirs)
        initializer.run()

        assert dirs[0].exists()
