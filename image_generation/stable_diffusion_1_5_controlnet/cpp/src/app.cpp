#include "gui.hpp"
#include <wx/dir.h>
#include <openvino/runtime/core.hpp>

bool ContainsModelFiles(const wxString& directory) {
    wxDir dir(directory);
    if (!dir.IsOpened()) {
        return false;
    }

    wxString filename;
    bool hasXml = dir.GetFirst(&filename, "*.xml", wxDIR_FILES);
    if (hasXml) {
        return true;
    }

    bool hasBin = dir.GetFirst(&filename, "*.bin", wxDIR_FILES);
    return hasBin;
}


std::vector<std::string> GetAvailableDevices() {
    ov::Core core;
    return core.get_available_devices();
}

AppFrame::AppFrame(const wxString& title) : wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, wxSize(800, 600)) {
    devices = GetAvailableDevices();
    notebook = new wxNotebook(this, wxID_ANY);


    InitMainPannel();
    InitEvents();


    SetClientSize(800, 600);
    SetSizer(new wxBoxSizer(wxVERTICAL));
    GetSizer()->Add(notebook, 1, wxEXPAND);

    Centre();
}

void AppFrame::InitMainPannel() {
    // Create a panel
    mainPanel = new wxPanel(notebook, wxID_ANY);
    notebook->AddPage(mainPanel, "Main");

    // Create a box sizer for the main layout
    wxBoxSizer* mainSizer = new wxBoxSizer(wxHORIZONTAL);

    // Create a vertical sizer for the left part
    wxBoxSizer* leftSizer = new wxBoxSizer(wxVERTICAL);

    // Create the image preview area
    leftSizer->Add(new wxStaticText(mainPanel, wxID_ANY, wxEmptyString), 0, wxALL, 5);
    inputImagePreview = new wxStaticBitmap(mainPanel, wxID_ANY, wxBitmap(200, 200));
    leftSizer->Add(inputImagePreview, 0, wxALL | wxALIGN_CENTER_HORIZONTAL, 5);

    // Create the image selection button
    selectImageButton = new wxButton(mainPanel, wxID_ANY, wxT("Select Image"));
    leftSizer->Add(selectImageButton, 0, wxALL | wxALIGN_CENTER_HORIZONTAL, 5);

    // Create the model selection button and text control
    wxBoxSizer* modelSizer = new wxBoxSizer(wxHORIZONTAL);
    modelSizer->Add(new wxStaticText(mainPanel, wxID_ANY, wxT("Model")), 0, wxALL | wxCENTER, 5);

    modelPathCtrl =
        new wxTextCtrl(mainPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize(300, -1), wxTE_READONLY);
    modelSizer->Add(modelPathCtrl, 1, wxALL | wxEXPAND, 5);

    selectModelButton = new wxButton(mainPanel, wxID_ANY, wxT("Select Model"));
    modelSizer->Add(selectModelButton, 0, wxALL | wxCENTER, 5);

    leftSizer->Add(modelSizer, 0, wxALL | wxEXPAND, 5);

    // Create the slider for steps
    wxBoxSizer* stepsSizer = new wxBoxSizer(wxHORIZONTAL);
    stepsSizer->Add(new wxStaticText(mainPanel, wxID_ANY, wxT("Steps")), 0, wxALL | wxCENTER, 5);

    // Create a text control to display and input the slider value
    stepsValueCtrl =
        new wxTextCtrl(mainPanel, wxID_ANY, wxT("20"), wxDefaultPosition, wxSize(50, -1), wxTE_PROCESS_ENTER);
    stepsSizer->Add(stepsValueCtrl, 0, wxALL | wxCENTER, 5);

    stepsSlider = new wxSlider(mainPanel, wxID_ANY, 20, 0, 50, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL);
    stepsSizer->Add(stepsSlider, 1, wxALL | wxEXPAND, 5);

    leftSizer->Add(stepsSizer, 0, wxALL | wxEXPAND, 5);

    // Create the spin control for seed
    wxBoxSizer* seedSizer = new wxBoxSizer(wxHORIZONTAL);
    seedSizer->Add(new wxStaticText(mainPanel, wxID_ANY, wxT("Seed")), 0, wxALL | wxCENTER, 5);
    seedSpinCtrl = new wxSpinCtrl(mainPanel,
                                    wxID_ANY,
                                    wxEmptyString,
                                    wxDefaultPosition,
                                    wxDefaultSize,
                                    wxSP_ARROW_KEYS,
                                    -1,
                                    2147483647,
                                    -1);
    seedSizer->Add(seedSpinCtrl, 1, wxALL | wxEXPAND, 5);
    leftSizer->Add(seedSizer, 0, wxALL | wxEXPAND, 5);

    // Create the device selection dropdown
    wxBoxSizer* deviceSizer = new wxBoxSizer(wxHORIZONTAL);
    deviceSizer->Add(new wxStaticText(mainPanel, wxID_ANY, wxT("Device")), 0, wxALL | wxCENTER, 5);

    deviceChoice = new wxChoice(mainPanel, wxID_ANY);
    for (const auto& device : devices) {
        deviceChoice->Append(device);
    }
    // Set a default selection if devices are available
    if (!devices.empty()) {
        deviceChoice->SetSelection(0);
    }
    deviceSizer->Add(deviceChoice, 1, wxALL | wxEXPAND, 5);
    leftSizer->Add(deviceSizer, 0, wxALL | wxEXPAND, 5);

    // Add new confirm button
    confirmButton = new wxButton(mainPanel, wxID_ANY, wxT("Confirm"));
    confirmButton->Disable();
    leftSizer->Add(confirmButton, 0, wxALL, 5);

    // Add leftSizer to the mainSizer
    mainSizer->Add(leftSizer, 1, wxALL | wxEXPAND, 5);

    // Create a vertical sizer for the right part
    wxBoxSizer* rightSizer = new wxBoxSizer(wxVERTICAL);

    // Create text areas for prompts
    promptTextCtrl = new wxTextCtrl(mainPanel, wxID_ANY, "Dancing Darth Vader, best quality, extremely detailed", wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE);
    rightSizer->Add(new wxStaticText(mainPanel, wxID_ANY, wxT("Prompt")), 0, wxALL, 5);
    rightSizer->Add(promptTextCtrl, 1, wxALL | wxEXPAND, 5);

    negativePromptTextCtrl = new wxTextCtrl(mainPanel, wxID_ANY, "monochrome, lowres, bad anatomy, worst quality, low quality", wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE);
    rightSizer->Add(new wxStaticText(mainPanel, wxID_ANY, wxT("Negative Prompt")), 0, wxALL, 5);
    rightSizer->Add(negativePromptTextCtrl, 1, wxALL | wxEXPAND, 5);

    // Add rightSizer to the mainSizer
    mainSizer->Add(rightSizer, 1, wxALL | wxEXPAND, 5);

    // Set the panel sizer
    mainPanel->SetSizer(mainSizer);
}

void AppFrame::OnSelectImage(wxCommandEvent& event) {
    wxFileDialog openFileDialog(this,
                                _("Open Image file"),
                                "",
                                "",
                                "Image files (*.jpg;*.png;*.bmp)|*.jpg;*.png;*.bmp",
                                wxFD_OPEN | wxFD_FILE_MUST_EXIST);

    if (openFileDialog.ShowModal() == wxID_CANCEL)
        return;

    wxString filePath = openFileDialog.GetPath();
    wxImage image;
    if (image.LoadFile(filePath)) {
        int previewWidth = inputImagePreview->GetSize().GetWidth();
        int previewHeight = inputImagePreview->GetSize().GetHeight();
        int imgWidth = image.GetWidth();
        int imgHeight = image.GetHeight();

        // Calculate the new width and height while maintaining aspect ratio
        double aspectRatio = static_cast<double>(imgWidth) / imgHeight;
        int newWidth, newHeight;
        if (previewWidth / static_cast<double>(previewHeight) > aspectRatio) {
            newHeight = previewHeight;
            newWidth = static_cast<int>(previewHeight * aspectRatio);
        } else {
            newWidth = previewWidth;
            newHeight = static_cast<int>(previewWidth / aspectRatio);
        }

        // Resize the image
        wxImage scaledImage = image.Scale(newWidth, newHeight, wxIMAGE_QUALITY_HIGH);

        // Clear the previous image and center the new image in the preview area
        wxBitmap previewBitmap(previewWidth, previewHeight);
        wxMemoryDC dc;
        dc.SelectObject(previewBitmap);
        dc.SetBackground(*wxWHITE_BRUSH);
        dc.Clear();
        dc.DrawBitmap(wxBitmap(scaledImage), (previewWidth - newWidth) / 2, (previewHeight - newHeight) / 2, true);
        dc.SelectObject(wxNullBitmap);

        inputImagePreview->SetBitmap(previewBitmap);
        Layout();  // Ensure the layout is updated
    }
}

void AppFrame::OnGenerate(wxCommandEvent& event) {
    wxDialog* resultDialog = new wxDialog(this, wxID_ANY, "Generated Image");
    wxStaticText* text = new wxStaticText(resultDialog, wxID_ANY, "Here would be the generated image");
    resultDialog->SetClientSize(text->GetBestSize());
    resultDialog->ShowModal();
    resultDialog->Destroy();
}

void AppFrame::ValidateSettings(wxCommandEvent& event) {
    bool valid = !modelPathCtrl->GetValue().IsEmpty() && stepsSlider->GetValue() >= 0 &&
                 seedSpinCtrl->GetValue() >= -1 && !promptTextCtrl->GetValue().IsEmpty();
    confirmButton->Enable(valid);
}

void AppFrame::InitEvents() {
    // select image
    selectImageButton->Bind(wxEVT_BUTTON, &AppFrame::OnSelectImage, this);

    // file dialog
    selectModelButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        wxDirDialog openDirDialog(this, _("Select Model Directory"), "", wxDD_DEFAULT_STYLE | wxDD_DIR_MUST_EXIST);

        if (openDirDialog.ShowModal() == wxID_CANCEL)
            return;

        wxString dirPath = openDirDialog.GetPath();
        if (ContainsModelFiles(dirPath)) {
            modelPathCtrl->SetValue(dirPath);
        } else {
            wxMessageBox("Selected directory does not contain any .xml or .bin files.", "Error", wxOK | wxICON_ERROR);
        }
    });

    // steps events to synchronize the slider and text control
    stepsSlider->Bind(wxEVT_SLIDER, [this](wxCommandEvent& event) {
        stepsValueCtrl->SetValue(wxString::Format("%d", event.GetInt()));
    });

    stepsValueCtrl->Bind(wxEVT_TEXT_ENTER, [this](wxCommandEvent& event) {
        long value;
        if (stepsValueCtrl->GetValue().ToLong(&value) && value >= 0 && value <= 50) {
            stepsSlider->SetValue(value);
        } else {
            stepsValueCtrl->SetValue(wxString::Format("%d", stepsSlider->GetValue()));
        }
    });

    stepsValueCtrl->Bind(wxEVT_KILL_FOCUS, [this](wxFocusEvent& event) {
        long value;
        if (stepsValueCtrl->GetValue().ToLong(&value) && value >= 0 && value <= 50) {
            stepsSlider->SetValue(value);
        } else {
            stepsValueCtrl->SetValue(wxString::Format("%d", stepsSlider->GetValue()));
        }
        event.Skip();
    });


    // settings validation
    modelPathCtrl->Bind(wxEVT_TEXT, [this](wxCommandEvent& evt) {
        ValidateSettings(evt);
    });
    stepsSlider->Bind(wxEVT_SLIDER, [this](wxCommandEvent& evt) {
        ValidateSettings(evt);
    });
    seedSpinCtrl->Bind(wxEVT_SPINCTRL, [this](wxCommandEvent& evt) {
        ValidateSettings(evt);
    });
    promptTextCtrl->Bind(wxEVT_TEXT, [this](wxCommandEvent& evt) {
        ValidateSettings(evt);
    });

    // start button
    confirmButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {});
}


class MyApp : public wxApp
{
public:
    bool OnInit() override
    {
        wxInitAllImageHandlers();
        AppFrame* frame = new AppFrame("Stable Diffusion Controlnet Demo");
        frame->Show(true);
        return true;
    }
};

wxIMPLEMENT_APP(MyApp);