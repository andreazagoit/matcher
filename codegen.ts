import { CodegenConfig } from "@graphql-codegen/cli";
import * as dotenv from "dotenv";
import * as path from "path";

dotenv.config({ path: path.resolve(__dirname, ".env") });

const config: CodegenConfig = {
    overwrite: true,
    // Uses introspection against the running dev server
    schema: `${process.env.NEXT_PUBLIC_APP_URL}/api/graphql`,
    // Only scan centralized gql.ts operation files
    documents: ["lib/models/**/gql.ts"],
    ignoreNoDocuments: true,
    generates: {
        "./lib/graphql/__generated__/graphql.ts": {
            plugins: ["typescript", "typescript-operations"],
            config: {
                // Use `null` for nullable fields instead of optionals
                avoidOptionals: {
                    field: true,
                    inputValue: false,
                },
                // Use `unknown` instead of `any` for unconfigured scalars
                defaultScalarType: "unknown",
                // Apollo Client always includes `__typename` fields
                nonOptionalTypename: true,
                // Don't generate `__typename` for root operation types
                skipTypeNameForRoot: true,
            },
        },
    },
};

export default config;
