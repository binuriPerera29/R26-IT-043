import { Routes, Route, Outlet } from "react-router-dom";
import DiabeticRetinopathy from "./components/DiabeticRetinopathy";
import OCTFundusAnalyser from "./components/OCTFundusAnalyser";
import GlaucomaFundusAnalyser from "./components/GlaucomaFundusAnalyser";
import VDSFundusAnalyser from "./components/VDSFundusAnalyser";
import Home from "./components/Home";
import NavBar from "./components/NavBar";
import Footer from "./components/Footer";

// Layout wrapper
function Layout() {
  return (
    <>
      <NavBar />
      <Outlet />
      <Footer />
    </>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/dr" element={<DiabeticRetinopathy />} />
        <Route path="/oct" element={<OCTFundusAnalyser />} />
        <Route path="/glaucoma" element={<GlaucomaFundusAnalyser />} />
        <Route path="/cataract" element={<VDSFundusAnalyser />} />
      </Route>
    </Routes>
  );
}
