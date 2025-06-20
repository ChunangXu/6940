import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Sidebar from "./components/sidebar/sidebar";
import Topbar from "./components/topbar/topbar";
import Dashboard from "./pages/Dashboard";
import IdentifyByPrompt from "./pages/IdentifyByPrompt";
import IdentifyByText from "./pages/IdentifyByText";
import "./App.css";

function App() {
  return (
    <Router>
      <Topbar />
      <div className="container">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/identify_by_prompt" element={<IdentifyByPrompt />} />
            <Route path="/identify_by_text" element={<IdentifyByText />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}