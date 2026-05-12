import Layout from '@theme/Layout';

import { FeaturesSection } from './_sections/FeaturesSection';
import { HeroSection } from './_sections/HeroSection';
import { InstallSection } from './_sections/InstallSection';
import { UseCasesSection } from './_sections/UseCasesSection';

export default function Home() {
  return (
    <Layout description="Run Generative AI models with simple C++/Python API and using OpenVINO Runtime">
      <HeroSection />
      <FeaturesSection />
      <UseCasesSection />
      <InstallSection />
    </Layout>
  );
}
