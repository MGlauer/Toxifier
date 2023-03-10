import * as React from 'react';
import { useEffect } from 'react';
import axios from "axios";
import PropTypes from 'prop-types';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Divider from '@mui/material/Divider';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import CloseIcon from '@mui/icons-material/Close';
import Chip from '@mui/material/Chip';
import DeleteIcon from '@mui/icons-material/DeleteOutlined';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import LightbulbIcon from '@mui/icons-material/Lightbulb';
import SaveIcon from '@mui/icons-material/Save';
import CancelIcon from '@mui/icons-material/Close';
import StartIcon from '@mui/icons-material/Start';
import Modal from '@mui/material/Modal';
import {
    GridRowModes,
    DataGrid,
    GridToolbarContainer,
    GridActionsCellItem,
} from '@mui/x-data-grid';
import {
    randomId,
} from '@mui/x-data-grid-generator';
import DetailsPage from "./details-page";

const RenderDate = (props) => {
  const { hasFocus, value } = props;
  const buttonElement = React.useRef(null);
  const rippleRef = React.useRef(null);

  React.useLayoutEffect(() => {
    if (hasFocus) {
      const input = buttonElement.current?.querySelector('input');
      input?.focus();
    } else if (rippleRef.current) {
      // Only available in @mui/material v5.4.1 or later
      rippleRef.current.stop({});
    }
  }, [hasFocus]);

  return (
    <strong>
      {value?.getFullYear() ?? ''}
      <Button
        component="button"
        ref={buttonElement}
        touchRippleRef={rippleRef}
        variant="contained"
        size="small"
        style={{ marginLeft: 16 }}
        // Remove button from tab sequence when cell does not have focus
        tabIndex={hasFocus ? 0 : -1}
        onKeyDown={(event) => {
          if (event.key === ' ') {
            // Prevent key navigation when focus is on button
            event.stopPropagation();
          }
        }}
      >
        Open
      </Button>
    </strong>
  );
};

function EditToolbar(props) {
    const {setRows, setRowModesModel, rows, getLabel} = props;

    const addRows = ((smiles) => {
            const ids = smiles.map((s) => randomId());
            setRows((oldRows) => [...oldRows, ...smiles.map((s, i) => ({id: ids[i], smiles: s, direct_parents: [], predicted_parents: []}))]);
            return ids
        }
    )

    const handleAdd = () => {
        const ids = addRows([''])
        setRowModesModel((oldModel) => ({
            ...oldModel,
            ...Object.fromEntries(ids.map(id => [id, {mode: GridRowModes.Edit, fieldToFocus: 'smiles'}])),
        }));
    };

    const handleUpload = (event) => {
        event.preventDefault();
        const reader = new FileReader()
        reader.onload = async (e) => {
            addRows(e.target.result.split("\n"))
        };
        reader.readAsText(event.target.files[0])
    };

    const handleDownload = (event) => {
        event.preventDefault();
        const fileData = JSON.stringify(rows.map((r) => ({"smiles": r["smiles"], "direct_parents": r["direct_parents"],"predicted_parents": r["predicted_parents"],})).filter((d) => d["direct_parents"].length !== 0));
        const blob = new Blob([fileData], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.download = "chebifier-predictions.json";
        link.href = url;
        link.click();
    };

    const handleRun = () => {
        console.log(rows);
        axios({
            url: '/api/classify',
            method: 'post',
            data: {smiles: rows.map((r) => (r["smiles"]))}
        }).then(response => {
            setRows((oldRows) => oldRows.map((row, i) => ({
                ...row, "direct_parents": response.data.direct_parents[i], "predicted_parents": response.data.predicted_parents[i],
            })));
        });
    };

    return (
        <GridToolbarContainer>
            <Button color="primary" startIcon={<AddIcon/>} onClick={handleAdd}>
                Add SMILES
            </Button>
            <Button color="primary" startIcon={<FileUploadIcon/>} component="label">
                Upload file
                <input
                    accept="text/plain"
                    style={{display: 'none'}}
                    id="file-upload"
                    type="file"
                    onChange={handleUpload}
                />
            </Button>
            <Divider/>
            <Button color="primary" startIcon={<StartIcon/>} onClick={handleRun}>
                Predict classes
            </Button>
            <Button color="primary" startIcon={<FileDownloadIcon/>} onClick={handleDownload} disabled={rows.filter((d) => d["direct_parents"].length !== 0).length === 0}>
                Download JSON
            </Button>
        </GridToolbarContainer>
    );
}

EditToolbar.propTypes = {
    setRowModesModel: PropTypes.func.isRequired,
    setRows: PropTypes.func.isRequired,
};

export default function ClassificationGrid() {
    const [rows, setRows] = React.useState([]);
    const [rowModesModel, setRowModesModel] = React.useState({});
    const [detail, setDetail] = React.useState(null);
    const [hierarchy, setHierarchy] = React.useState({});

    if (Object.keys(hierarchy).length === 0) {
        axios.get('/api/hierarchy').then(response => {
            setHierarchy(response.data);
        });
    }


    const handleRowEditStart = (params, event) => {
        event.defaultMuiPrevented = true;
    };

    const handleRowEditStop = (params, event) => {
        event.defaultMuiPrevented = true;
    };

    const handleEditClick = (id) => () => {
        setRowModesModel({...rowModesModel, [id]: {mode: GridRowModes.Edit}});
    };

    const handleSaveClick = (id) => () => {
        setRowModesModel({...rowModesModel, [id]: {mode: GridRowModes.View}});
    };

    const handleDeleteClick = (id) => () => {
        setRows(rows.filter((row) => row.id !== id));
    };

    const handleCancelClick = (id) => () => {
        setRowModesModel({
            ...rowModesModel,
            [id]: {mode: GridRowModes.View, ignoreModifications: true},
        });

        const editedRow = rows.find((row) => row.id === id);
        if (editedRow.isNew) {
            setRows(rows.filter((row) => row.id !== id));
        }
    };

    const processRowUpdate = (newRow) => {
        const updatedRow = {...newRow, isNew: false};
        setRows(rows.map((row) => (row.id === newRow.id ? updatedRow : row)));
        return updatedRow;
    };

    const renderClasses = (params) => {
        const data = params.value;
        if (data == null){
          return  <Typography>Could not process input</Typography>
        } else {
            return <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
                {(data.map((x) => <Chip component="a" href={"http://purl.obolibrary.org/obo/" + x.replace(":", "_")} label={hierarchy[x].label} clickable target="_blank"/>))}
                </Box>
        }
    };

    const [open, setOpen] = React.useState(false);
    const handleOpen = (id) => () => {
        const thisRow = rows.find((row) => row.id === id);
        axios.post('/api/details', {smiles: thisRow.smiles}).then(response => {
            setDetail({
                plain_molecule: response.data.figures.plain_molecule,
                graphs: response.data.graphs,
                chebi: response.data.classification
            });
            setOpen(true);
        });

    }
    const handleClose = () => setOpen(false);

    const columns = [
        {field: 'smiles', headerName: 'Smiles', flex: 0.45, editable: true},
        {field: 'direct_parents', headerName: 'Predicted Class', flex: 0.45, editable: false, renderCell:renderClasses},
        {
            field: 'actions',
            type: 'actions',
            headerName: 'Actions',
            flex: 0.1,
            cellClassName: 'actions',
            getActions: ({id}) => {
                const isInEditMode = rowModesModel[id]?.mode === GridRowModes.Edit;
                const thisRow = rows.find((row) => row.id === id);
                const wasPredicted = thisRow.direct_parents.length > 0;

                if (isInEditMode) {
                    return [
                        <GridActionsCellItem
                            icon={<SaveIcon/>}
                            label="Save"
                            onClick={handleSaveClick(id)}
                        />,
                        <GridActionsCellItem
                            icon={<CancelIcon/>}
                            label="Cancel"
                            className="textPrimary"
                            onClick={handleCancelClick(id)}
                            color="inherit"
                        />,
                    ];
                }

                return [
                    <GridActionsCellItem
                        icon={<EditIcon/>}
                        label="Edit"
                        className="textPrimary"
                        onClick={handleEditClick(id)}
                        color="inherit"
                    />,
                    <GridActionsCellItem
                        icon={<DeleteIcon/>}
                        label="Delete"
                        onClick={handleDeleteClick(id)}
                        color="inherit"
                    />,
                    <GridActionsCellItem
                        icon={<LightbulbIcon/>}
                        label="Details"
                        onClick={handleOpen(id)}
                        color="inherit"
                        disabled={!wasPredicted}
                    />,
                ];
            },
        },
    ];

    const getLabel = (x) => {
        return hierarchy[x]["label"]
    }

    return (
        <Box sx={{width: "100%"}}>
            <Box>
                <h1>Chebifier</h1>
                Classify chemical structures using AI.
            </Box>
            <Paper sx={{width: "100%"}}>
                <Box
                    sx={{
                        height: 500,
                        width: '100%',
                        '& .actions': {
                            color: 'text.secondary',
                        },
                        '& .textPrimary': {
                            color: 'text.primary',
                        },
                    }}
                >
                    <DataGrid
                        rows={rows}
                        columns={columns}
                        editMode="row"
                        rowModesModel={rowModesModel}
                        onRowModesModelChange={(newModel) => setRowModesModel(newModel)}
                        onRowEditStart={handleRowEditStart}
                        onRowEditStop={handleRowEditStop}
                        processRowUpdate={processRowUpdate}
                        getRowHeight={() => 'auto'}
                        components={{
                            Toolbar: EditToolbar,
                        }}
                        componentsProps={{
                            toolbar: {setRows, setRowModesModel, rows, getLabel},
                        }}
                        experimentalFeatures={{newEditingApi: true}}
                    />
                </Box>
            </Paper>

            <Modal
              open={open}
              onClose={handleClose}
              aria-labelledby="modal-modal-title"
              aria-describedby="modal-modal-description"
            >
                <Box sx={{
                  mb: 2,
                  display: "flex",
                  flexDirection: "column",
                  height: 700,
                  overflow: "hidden",
                  overflowY: "scroll",
                }}>
                    <Button color="primary" onClick={handleClose} startIcon={<CancelIcon/>}/>
                    <DetailsPage detail={detail} />
                </Box>
            </Modal>

        </Box>
    );
}
