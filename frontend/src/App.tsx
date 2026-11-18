import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Navigation } from "./components/Navigation";
import { EvaluationDashboardPage } from "./pages/EvaluationDashboardPage";
import { ResearchAssistantPage } from "./pages/ResearchAssistantPage";
export default function App() { return <BrowserRouter><Navigation/><Routes><Route path="/" element={<ResearchAssistantPage/>}/><Route path="/dashboard" element={<EvaluationDashboardPage/>}/></Routes><footer><span>Evidence first · Transparent by design</span><span>AI Powered Research Assistant</span></footer></BrowserRouter>; }
