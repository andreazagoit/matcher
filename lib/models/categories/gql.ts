import { gql } from "graphql-tag";
import { EVENT_CARD_FRAGMENT } from "@/lib/models/events/gql";
import { SPACE_FRAGMENT } from "@/lib/models/spaces/gql";

export const GET_CATEGORIES = gql`
  query GetCategories {
    categories {
      id
    }
  }
`;

export const GET_CATEGORY = gql`
  ${EVENT_CARD_FRAGMENT}
  ${SPACE_FRAGMENT}
  query GetCategory($id: ID!, $eventsLimit: Int, $spacesLimit: Int) {
    category(id: $id) {
      id
      recommendedEvents(limit: $eventsLimit) {
        ...EventCardFields
      }
      recommendedSpaces(limit: $spacesLimit) {
        ...SpaceFields
      }
      recommendedCategories(limit: 6)
    }
  }
`;
