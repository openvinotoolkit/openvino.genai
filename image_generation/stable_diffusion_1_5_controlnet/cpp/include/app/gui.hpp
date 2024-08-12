// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <wx/wx.h>
#include <wx/sizer.h>
#include <wx/filedlg.h>
#include <wx/image.h>
#include <wx/statbmp.h>
#include <wx/slider.h>
#include <wx/spinctrl.h>
#include <wx/notebook.h>
#include <vector>
#include <string>
#include "worker.hpp"
#include "core/core.hpp"

 class AppFrame : public wxFrame {
 public:
     AppFrame(const wxString& title);
     ImageToImagePipeline* GetImageToImagePipeline();

 private:
     void InitMainPannel();
     void InitEvents();
     void InitWorkers();

     void OnSelectImage(wxCommandEvent& event);
     void OnGenerate(wxCommandEvent& event);
     void ValidateSettings(wxCommandEvent& event);
     void GetImageToImageParam(StableDiffusionControlnetPipelineParam& param);

    void OnImageGenCompleted(wxThreadEvent& event);
     
     wxNotebook* notebook;

     wxPanel* mainPanel;
     wxButton* selectImageButton;
     wxStaticBitmap* inputImagePreview;
     wxTextCtrl* modelPathCtrl;
     wxButton* selectModelButton;
     wxTextCtrl* promptTextCtrl;
     wxTextCtrl* negativePromptTextCtrl;
     wxSlider* stepsSlider;
     wxTextCtrl* stepsValueCtrl;
     wxSpinCtrl* seedSpinCtrl;
     wxButton* confirmButton;
     wxChoice* deviceChoice;

     std::vector<std::string> devices;
     std::string inputImagePath;

     std::string currentModelPath;
     std::string currentDevice;

     WorkerThread* workerThread;
     ImageToImagePipeline *imageToImagePipeline;
 };
