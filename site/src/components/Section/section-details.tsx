import {SectionTitle} from "./section-title";
import {SectionDescription} from "./section-description";
import {SectionFeatures} from "./section-features";
import {GoToDocumentation} from "@site/src/components/GoToLink/go-to-documentation";
import {ExploreCodeSamples} from "@site/src/components/GoToLink/explore-code-samples";
import {LanguageTabs} from "@site/src/components/LanguageTabs/language-tabs";
import {ComponentProps} from "react";

interface SectionDescriptionProps extends ComponentProps<typeof LanguageTabs>{
    title: string;
    description: string;
    features: (string | JSX.Element)[];
    documentationLink: string;
    codeSamplesLink: string;
}

export const SectionDetails = ({ title, description, features, documentationLink, codeSamplesLink, items, selectedTab }: SectionDescriptionProps) => {
    return (
        <div>
            <div>
                <SectionTitle>{title}</SectionTitle>
                <SectionDescription>{description}</SectionDescription>
            </div>

            <hr/>

            <SectionFeatures features={features} />

            <hr/>

            <LanguageTabs items={items} selectedTab={selectedTab} />

            <hr/>

            <ExploreCodeSamples link={codeSamplesLink} />
            <GoToDocumentation link={documentationLink} />
        </div>
    )
}