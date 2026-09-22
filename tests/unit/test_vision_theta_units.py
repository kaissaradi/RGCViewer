import numpy as np

from src.analysis.visionloader import VisionCellDataTable, VisionFieldNames


def test_vision_theta_is_exposed_in_radians_without_conversion():
    table = VisionCellDataTable()
    table.main_datatable[1] = {
        VisionFieldNames.STA_XCENTER_FIELDNAME: 10.0,
        VisionFieldNames.STA_YCENTER_FIELDNAME: 20.0,
        VisionFieldNames.STA_FITX_STD_FIELDNAME: 2.0,
        VisionFieldNames.STA_FITY_STD_FIELDNAME: 1.0,
        VisionFieldNames.STA_ROT_FIELDNAME: np.pi / 2.0,
    }

    fit = table.get_stafit_for_cell(1)

    assert fit.rot == np.pi / 2.0

