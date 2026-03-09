import { gql } from "graphql-tag";

export const PROFILE_ITEM_FRAGMENT = gql`
  fragment UserItemFields on UserItem {
    id
    userId
    type
    promptKey
    content
    displayOrder
    createdAt
    updatedAt
  }
`;
