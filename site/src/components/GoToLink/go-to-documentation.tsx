import {GoToLink} from "@site/src/components/GoToLink/go-to-link";
import {FC} from "react";

type GoToDocumentationProps = {
    link: string;
}

export const GoToDocumentation: FC<GoToDocumentationProps> = ({link}) => {
    return (
        <GoToLink link={link} name={'Go to Documentation'} />
    )
}