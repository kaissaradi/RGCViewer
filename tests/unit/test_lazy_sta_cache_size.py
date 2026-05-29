from unittest.mock import MagicMock, patch


def test_lazy_sta_cache_size_scales_with_dataset(tmp_path):
    from src.analysis.vision_integration import LazySTADict, MAX_STA_CACHE_CELLS

    mock_reader = MagicMock()

    mock_reader.cell_id_to_byte_offset = {i: i * 100 for i in range(800)}
    with patch('src.analysis.vision_integration.vl') as mock_vl:
        mock_vl.STAReader.return_value = mock_reader
        lazy = LazySTADict(tmp_path, 'test_dataset')

    assert lazy._max_cache == min(MAX_STA_CACHE_CELLS, max(200, 800))

    mock_reader.cell_id_to_byte_offset = {i: i * 100 for i in range(150)}
    with patch('src.analysis.vision_integration.vl') as mock_vl:
        mock_vl.STAReader.return_value = mock_reader
        lazy_small = LazySTADict(tmp_path, 'test_dataset')

    assert lazy_small._max_cache == min(MAX_STA_CACHE_CELLS, max(200, 150))
    assert lazy_small._max_cache >= 200


def test_max_sta_cache_cells_constant_exists():
    from src.analysis.vision_integration import MAX_STA_CACHE_CELLS

    assert isinstance(MAX_STA_CACHE_CELLS, int)
    assert MAX_STA_CACHE_CELLS >= 200
