import { gql } from "graphql-tag";

export const UPDATE_MY_INTERESTS = gql`
  mutation UpdateMyInterests($tags: [String!]!) {
    updateMyInterests(tags: $tags) {
      tag
      weight
    }
  }
`;
